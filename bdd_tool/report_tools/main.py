# main.py
import os
import config
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from PIL import ExifTags

# 导入模块
from loaders.yolo_loader import YoloLoader
from exporters.pdf_exporter import PDFExporter
from utils.visualization import draw_box, crop_box
from utils.analysis import level_judge, action_judge, img_sta
# main.py 顶部 import 部分
from exporters.pdf_styles import *
from utils.exif_dji_xmp import parse_dji_xmp
from utils.geo_utils import calculate_floor, calculate_orientation, calculate_gsd, pixel_to_physical
from utils.exif_dji_xmp import parse_dji_xmp


# 定义样式映射表
EXPORTER_MAP = {
    0: PDFExporterBasic,
    1: PDFExporterDetailed,
    2: PDFExporterMeasurement,
    3: PDFExporterCompact,
}

def safe_float(value):
    """
    处理 '6.7 mm', '4.67 m', None, 0 等各种情况转 float
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    
    val_str = str(value).strip()
    try:
        return float(val_str)
    except ValueError:
        # 提取数字部分 (支持 "6.7 mm", "+93.9 m")
        match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
        if match:
            return float(match.group())
        return 0.0

class ReportEngine:
    def __init__(self, loader, labels, metadata_getter=None):
        """
        :param metadata_getter: 一个函数，接收 img_path，返回包含 xyz, orientation, floor, view 的字典
        """
        self.loader = loader
        self.labels = labels
        self.colors = config.COLOR_PALETTE
        self.metadata_getter = metadata_getter  # <--- 新增接口
        


        self._init_metadata_store()

    def _init_metadata_store(self):
        """
        初始化或重置全局元数据容器。
        支持连续运行多组数据时清理上一组的状态。
        """
        self.global_drone_info = {}

        # 临时路径
        self.base_dir = None
        self.vis_dir = None
        self.crop_dir = None

    def _get_unified_metadata(self, img_path, img_pil):
        """
        【修改点】在返回的 meta 中增加 '_parsing_method' 字段
        """
        meta = {}
        method_used = "Unknown"

        # --- 策略 1: 尝试 PyExif ---
        try:
            import pyexif
            exif_editor = pyexif.ExifEditor(str(img_path))
            meta = exif_editor.getDictTags()
            method_used = "PyExif (ExifTool)"
        except (ImportError, FileNotFoundError, Exception):
            # --- 策略 2: Fallback ---
            method_used = "Fallback (PIL + XMP)"
            
            # A. 标准 EXIF
            std_exif = self._get_standard_exif(img_pil)
            meta.update(std_exif)
            
            # B. DJI XMP
            xmp_data = parse_dji_xmp(img_path)
            meta.update(xmp_data)

        # --- 3. 标准化 ---
        if 'Model' not in meta and 'DroneModel' in meta:
            meta['Model'] = meta['DroneModel']

        # 【新增】记录解析方式，方便日志打印
        meta['_parsing_method'] = method_used
        
        return meta

    def _load_global_metadata(self, first_img_path):
        """
        【修改点】增强日志输出，显示解析方式和镜头物理参数
        """
        try:
            img = Image.open(first_img_path)
            all_meta = self._get_unified_metadata(first_img_path, img)
            stem_name = Path(first_img_path).stem
            
            # 1. 提取基础信息
            self.global_drone_info = {
                'Model': all_meta.get('Model', 'Unknown'),
                'Camera': all_meta.get('ImageSource', 'Unknown'),
                'Firmware': all_meta.get('Firmware', all_meta.get('Version', 'Unknown'))
            }

            # 2. 计算物理参数 (用于展示)
            specs, _ = self._get_camera_specs_unified(all_meta, stem_name)
            sensor_width = specs['sensor_width_mm']
            
            # 判定焦距来源
            raw_focal = safe_float(all_meta.get('FocalLength'))
            if raw_focal > 0:
                actual_focal = raw_focal
                focal_source = "Exif Data"
            else:
                actual_focal = specs['focal_length_mm']
                focal_source = "Config Default"

            # 3. 打印增强版日志
            print("\n" + "="*50)
            print(f"--- [Metadata Declaration] ---")
            print(f" Source Image   : {Path(first_img_path).name}")
            print(f" Parsing Method : {all_meta.get('_parsing_method', 'Unknown')}")
            print("-" * 50)
            print(f" Drone Model    : {self.global_drone_info['Model']}")
            print(f" Camera Source  : {self.global_drone_info['Camera']}")
            print(f" Firmware Ver   : {self.global_drone_info['Firmware']}")
            print("-" * 50)
            print(f" [Lens Physics] (Used for GSD Calculation)")
            print(f" Sensor Width   : {sensor_width} mm")
            print(f" Focal Length   : {actual_focal} mm (Source: {focal_source})")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Warning: Failed to load global metadata from {first_img_path}: {e}")
            self.global_drone_info = {'Model': 'Unknown', 'Camera': 'Unknown'}

    def _get_standard_exif(self, img):
        """[Fallback组件] 使用 PIL 获取标准 EXIF，尝试读取 SubIFD 以获取焦距"""
        exif_data = {}
        try:
            # 1. 基础 EXIF
            info = img.getexif()
            if info:
                # 272: Model
                if 272 in info: 
                    exif_data['Model'] = str(info[272]).strip()
                # 37386: FocalLength (有时在主 IFD)
                if 37386 in info:
                    exif_data['FocalLength'] = float(info[37386])

                # 2. 尝试读取 Exif SubIFD (0x8769 = 34665)
                # 很多相机的焦距藏在这里
                if 34665 in info:
                    sub_ifd = info.get_ifd(34665)
                    if 37386 in sub_ifd:
                        exif_data['FocalLength'] = float(sub_ifd[37386])
        except Exception:
            pass
        return exif_data

    def _get_camera_specs_unified(self, meta_dict, filename):
        """根据元数据字典获取 config 参数"""
        # 1. 确定型号 (兼容 pyexif 的 'Model' 和 XMP 的 'DroneModel')
        model = meta_dict.get('Model') or meta_dict.get('DroneModel')
        
        # 清洗型号字符串
        if model:
            model_str = str(model)
            if 'Matrice 4' in model_str or 'M4' in model_str: model = 'M4T'
            elif 'Mavic 3 Thermal' in model_str: model = 'M3T'
            elif 'Mavic 3 Enterprise' in model_str: model = 'M3E'
            elif 'Matrice 30' in model_str: model = 'M30T'
        else:
            model = 'Unknown'

        # 2. 确定类型 (Wide vs Thermal)
        img_source = str(meta_dict.get('ImageSource', ''))
        is_thermal = '_T' in filename or 'Thermal' in img_source or 'IR' in img_source
        cam_type = 'Thermal' if is_thermal else 'Wide'
        
        # 3. 查表
        config_key = f"{model}_{cam_type}"
        specs = config.DRONE_PARAMS.get(config_key, config.DRONE_PARAMS['default'])
        
        return specs

    def _get_camera_specs_unified(self, meta_dict, filename):
        """
        统一接口：根据元数据字典确定相机参数
        """
        # 1. 确定型号
        model = meta_dict.get('Model') or meta_dict.get('DroneModel')
        
        # 清洗型号名称
        if model:
            model_str = str(model)
            if 'Matrice 4' in model_str or 'M4' in model_str: model = 'M4T'
            elif 'Mavic 3 Thermal' in model_str: model = 'M3T'
            elif 'Mavic 3 Enterprise' in model_str: model = 'M3E'
            elif 'Matrice 30' in model_str: model = 'M30T'
        else:
            model = 'Unknown'

        # 2. 确定类型 (Visible vs Thermal)
        # 检查文件名或 ImageSource
        img_source = str(meta_dict.get('ImageSource', ''))
        is_thermal = '_T' in filename or 'Thermal' in img_source or 'IR' in img_source
        cam_type = 'Thermal' if is_thermal else 'Wide'
        
        # 3. 查表
        config_key = f"{model}_{cam_type}"
        specs = config.DRONE_PARAMS.get(config_key, config.DRONE_PARAMS['default'])
        
        return specs, model
    def _process_single_image(self, img_path, detections):
        stem_name = Path(img_path).stem
        os.makedirs(self.vis_dir, exist_ok=True)
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)

        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # --- 1. 【核心修改】统一获取所有元数据 ---
        # 替代了之前的 _get_standard_exif 和 parse_dji_xmp
        all_meta = self._get_unified_metadata(img_path, img)

        # ==========================================
        # 2. 计算物理参数 (使用统一后的 meta)
        # ==========================================
        # A. 获取传感器规格
        specs, detected_model = self._get_camera_specs_unified(all_meta, stem_name)
        sensor_width = specs['sensor_width_mm']

        # B. 获取焦距 (EXIF 优先 -> Config 兜底)
        focal_length = safe_float(all_meta.get('FocalLength'))
        if focal_length == 0:
            focal_length = specs['focal_length_mm']

        # C. 获取距离 (LRF 优先 -> 相对高度 -> Config 兜底)
        distance_mm = 0
        
        # 尝试读取激光测距 (兼容 pyexif 的 '4.67 m' 和 XMP 的 '4.67')
        lrf = all_meta.get('LRFTargetDistance')
        if lrf:
            val = safe_float(lrf)
            if val > 0: distance_mm = val * 1000
            
        # 尝试相对高度
        if distance_mm == 0:
            rel_alt = all_meta.get('RelativeAltitude')
            if rel_alt:
                val = safe_float(rel_alt)
                if val != 0: distance_mm = abs(val) * 1000
        
        # Config 默认值兜底
        if distance_mm == 0:
            distance_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000

        # D. 计算 GSD
        gsd = calculate_gsd(
            distance_mm=distance_mm,
            focal_length_mm=focal_length,
            sensor_width_mm=sensor_width,
            image_width_pix=img_w
        )

        # 获取额外的显示用元数据 (位置、楼层等)
        # 依然可以使用自定义 getter 或者从 all_meta 提取
        img_meta = self._get_metadata(img_path, all_meta)

        # --- 2. 可视化绘制 ---
        img_vis = img.copy()
        if len(detections) > 0:
            img_vis = draw_box(img_vis, detections, self.labels, self.colors)
        vis_path = os.path.join(self.vis_dir, stem_name + '.png')
        img_vis.save(vis_path)

        # --- 3. 处理检测框与尺寸 ---
        records = []
        crops = crop_box(img, detections)
        
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            score = float(bbox[1])
            # 像素坐标
            x1, y1, x2, y2 = bbox[2:]
            
            # --- 【关键】计算尺寸字段 ---
            w_pix = x2 - x1
            h_pix = y2 - y1
            area_pix = w_pix * h_pix
            
            # 物理尺寸 (cm)
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)
            area_cm2 = (w_cm * h_cm) if (w_cm and h_cm) else None

            cat_name = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge(bbox[2:])
            
            crop_filename = f"{stem_name}_{i}_{cls_id}.png"
            crop_path = os.path.join(crop_subdir, crop_filename)
            crops[i].save(crop_path)

            record = {
                'Path': img_path,
                'VisPath': vis_path,
                'CropPath': crop_path,
                'Category': cat_name.title(),
                'Level': level,
                'Score': score,
                'Action': action_judge(level),
                # --- 【关键】写入这些字段 ---
                'W_pix': int(w_pix),
                'H_pix': int(h_pix),
                'Area_pix': int(area_pix),
                'W_cm': f"{w_cm:.1f}" if w_cm else "N/A",
                'H_cm': f"{h_cm:.1f}" if h_cm else "N/A",
                'Area_cm2': f"{area_cm2:.1f}" if area_cm2 else "N/A"
            }
            # 合并 img_meta
            record.update(img_meta)
            records.append(record)
        
        return pd.DataFrame(records)

    def _get_metadata(self, img_path, meta_dict=None):
        """
        获取楼层、XYZ等展示信息。
        增加了 meta_dict 参数，如果有现成的元数据就直接用，不用再读文件。
        """
        default_meta = {'xyz': 'None', 'orientation': 'None', 'floor': 'None', 'view': 'None'}
        
        if self.metadata_getter:
            try:
                # 稍微修改 getter 的调用约定，或者让 getter 自己决定怎么读
                # 为了兼容性，这里还是传 img_path，但如果你的 provider 支持接收 dict 更好
                external_meta = self.metadata_getter(img_path) 
                if external_meta:
                    default_meta.update(external_meta)
            except Exception as e:
                print(f"Warning: Metadata getter error: {e}")
        
        return default_meta
    def run(self, output_path, model_name="Generic-Model", style_id=0):
        # 初始化目录
        if self.base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(output_path))
            self.vis_dir = os.path.join(self.base_dir, 'report_vis')
            self.crop_dir = os.path.join(self.base_dir, 'report_crop')

        print("--- Loading Data ---")
        raw_data = self.loader.load()
        
        print("--- Processing Images ---")
        all_results_dfs = []
        img_paths_list = []
        
        # --- 【修改点3】元数据预加载逻辑 ---
        # 如果当前元数据为空，且数据列表不为空，则使用第一组数据初始化
        if not self.global_drone_info and raw_data:
            # 获取第一个数据单元 (YoloLoader 返回的是 dict list)
            first_item = raw_data[0]
            first_img_path = first_item['image_path']
            
            # 未来如果是 RGB+T 对，loader 可能会返回 {'rgb': '...', 'thermal': '...'}
            # 这里可以预留逻辑：if isinstance(first_img_path, dict): ...
            
            self._load_global_metadata(first_img_path)

        for item in tqdm(raw_data, desc="Visualizing"):
            img_path = item['image_path']
            dets = item['detections']
            img_paths_list.append(img_path)
            
            df = self._process_single_image(img_path, dets)
            all_results_dfs.append(df)

        # 统计逻辑
        total = len(raw_data)
        has_defect = sum(1 for df in all_results_dfs if not df.empty)
        cat_counts = {}
        for df in all_results_dfs:
            if not df.empty:
                counts = df['Category'].value_counts()
                for cat, count in counts.items():
                    cat_counts[cat] = cat_counts.get(cat, 0) + count

        report_info = {
            'input': {
                'number': total,
                'shape': img_sta(img_paths_list),
                'type': 'Images'
            },
            'output': {
                'model': model_name,
                'defects': has_defect,
                'no defects': total - has_defect,
                'defects sta': cat_counts
            },
            'records': all_results_dfs,
            'drone_info': self.global_drone_info,
        }

        # 3. 动态选择 Exporter
        print(f"--- Exporting Report using Style {style_id} ---")
        
        if style_id not in EXPORTER_MAP:
            print(f"Warning: Style {style_id} not found, defaulting to Basic (0).")
            style_id = 0
            
        # 实例化对应的类
        ExporterClass = EXPORTER_MAP[style_id]
        exporter_instance = ExporterClass()
        
        # 执行导出
        exporter_instance.export(report_info, output_path)

        # --- 【修改点4】运行结束后重置元数据 ---
        # 方便同一个 engine 实例被用于下一次 run 调用
        print("--- Resetting Metadata for next run ---")
        self._init_metadata_store()
def load_class_list(class_path):
    with open(class_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def my_metadata_provider(img_path):
    """
    用户自定义的接口。
    你可以从这里读取 CSV/Excel/Database，根据文件名找到对应信息。
    """
    # 示例：假设文件名包含楼层信息，例如 "IMG_F3_View1.jpg"
    # return {
    #     'xyz': '100,200,50',
    #     'orientation': 'North',
    #     'floor': '3F',
    #     'view': 'Front'
    # }
    return None  # 目前返回 None，ReportEngine 会使用默认值 'None'

def dji_metadata_provider(img_path):
    """
    使用 DJI XMP 解析图片元数据，并计算 XYZ、楼层和朝向。
    """
    # 1. 解析原始 XMP 数据
    xmp_data = parse_dji_xmp(img_path)
    if not xmp_data:
        return None

    # 2. 提取关键字段
    # 注意：XMP解析出来的是字符串，可能带 '+' 号
    lat = xmp_data.get('GpsLatitude', '0')
    lon = xmp_data.get('GpsLongitude', '0')
    abs_alt = xmp_data.get('AbsoluteAltitude', '0')
    rel_alt = xmp_data.get('RelativeAltitude', '0')
    yaw = xmp_data.get('GimbalYawDegree', '0')

    # 3. 组合 XYZ (这里选择使用绝对高度作为 Z，或者根据需求改为 RelAlt)
    # 通常 GPS 坐标记录的是 Lat, Lon, AbsAlt
    xyz_str = f"{lat}, {lon}, {abs_alt}"

    # 4. 计算楼层 (使用相对高度 RelativeAltitude)
    floor_str = calculate_floor(rel_alt, config.FLOOR_CONFIG)

    # 5. 计算朝向 (使用 GimbalYawDegree)
    # 注意：这里返回的是立面朝向 (View)，例如 "East" 表示东立面
    orientation_str = calculate_orientation(yaw)
    
    # 假设 View 字段就是朝向 + Elevation 的组合，或者保持 View 为纯方向
    view_str = f"{orientation_str} Elevation"

    return {
        'xyz': xyz_str,
        'orientation': orientation_str,
        'floor': floor_str,
        'view': view_str
    }

# main.py

# ... (Imports 保持不变，但可以删除 utils.exif_dji_xmp 的引用) ...
import re

def pyexif_to_dict(img_path):
    """
    使用 pyexif (ExifTool) 读取所有元数据
    """
    try:
        import pyexif
        # 注意：确保系统路径中有 exiftool，或者 pyexif 配置正确
        img = pyexif.ExifEditor(img_path)
        return img.getDictTags()
    except Exception as e:
        print(f"Error reading EXIF with pyexif: {e}")
        return {}

def safe_float(value):
    """
    安全转换为浮点数，自动处理 '6.7 mm' 或 'None' 等情况
    """
    if value is None:
        return 0.0
    
    # 如果已经是数字，直接返回
    if isinstance(value, (int, float)):
        return float(value)
    
    # 如果是字符串，尝试清洗
    value_str = str(value).strip()
    try:
        # 尝试直接转换
        return float(value_str)
    except ValueError:
        # 提取字符串中的第一个数字部分 (支持负号和小数点)
        # 例如: "6.7 mm" -> 6.7, "+93.9 m" -> 93.9
        match = re.search(r"[-+]?\d*\.\d+|\d+", value_str)
        if match:
            return float(match.group())
        return 0.0

if __name__ == '__main__':
    # 配置路径
    ROOT_DIR = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12'
    IMG_DIR = os.path.join(ROOT_DIR, 'val', 'images')
    PRED_DIR = os.path.join(ROOT_DIR, 'result_analysis', 'val_infer', 'labels')
    CLASS_PATH = os.path.join(ROOT_DIR, 'classes.txt')
    OUTPUT_PATH = os.path.join(ROOT_DIR, 'result_analysis', 'val_infer', 'report_modular.pdf')

    # 1. 准备组件
    classes = load_class_list(CLASS_PATH)
    
    # 这里可以轻松替换成 CocoLoader 或者 HTMLExporter
    my_loader = YoloLoader(img_dir=IMG_DIR, txt_dir=PRED_DIR)
    my_exporter = PDFExporter()

    # 2. 初始化引擎并运行
    engine = ReportEngine(loader=my_loader, labels=classes, metadata_getter=dji_metadata_provider)
    engine.run(OUTPUT_PATH, model_name="BDD-MODEL", style_id=3)