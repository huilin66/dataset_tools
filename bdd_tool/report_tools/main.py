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
    2: PDFExporterMeasurement # <--- 新增样式 3
}


class ReportEngine:
    def __init__(self, loader, labels, metadata_getter=None):
        """
        :param metadata_getter: 一个函数，接收 img_path，返回包含 xyz, orientation, floor, view 的字典
        """
        self.loader = loader
        self.labels = labels
        self.colors = config.COLOR_PALETTE
        self.metadata_getter = metadata_getter  # <--- 新增接口
        
        # 临时路径
        self.vis_dir = None
        self.crop_dir = None

        self.global_drone_info = {}
    def _get_standard_exif(self, img):
        """获取标准 EXIF (如焦距)"""
        exif_data = {}
        try:
            info = img.getexif()
            if info:
                for tag, value in info.items():
                    decoded = ExifTags.TAGS.get(tag, tag)
                    exif_data[decoded] = value
        except Exception:
            pass
        return exif_data

    def _get_metadata(self, img_path):
        """
        内部方法：获取元数据，如果没有提供 getter 或获取失败，返回默认 None
        """
        default_meta = {
            'xyz': 'None', 
            'orientation': 'None', 
            'floor': 'None', 
            'view': 'None'
        }
        
        if self.metadata_getter:
            try:
                # 调用外部传入的函数获取真实信息
                external_meta = self.metadata_getter(img_path)
                if external_meta:
                    default_meta.update(external_meta)
            except Exception as e:
                print(f"Warning: Failed to get metadata for {img_path}: {e}")
        
        return default_meta
    def _process_single_image(self, img_path, detections):
        """【核心修改】在此处计算 W_pix, W_cm 等字段"""
        stem_name = Path(img_path).stem
        os.makedirs(self.vis_dir, exist_ok=True)
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)

        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # --- 1. 计算 GSD 和获取元数据 ---
        # 获取标准 EXIF (焦距)
        exif_info = self._get_standard_exif(img)
        focal_length = exif_info.get('FocalLength', None)

        # 【新增】如果读不到焦距，或者焦距为0，强制使用默认值
        if focal_length is None or float(focal_length) == 0:
            # print(f"DEBUG: No FocalLength for {stem_name}, using default.")
            focal_length = getattr(config, 'DEFAULT_FOCAL_LENGTH_MM', 4.5)

        # 获取 DJI XMP 信息 (用于 GSD 计算)
        xmp_data = parse_dji_xmp(img_path)

        # print(f"DEBUG: {stem_name} XMP: {xmp_data.get('LRFTargetDistance')}")

        # --- 计算距离 (优化版逻辑) ---
        distance_mm = 0
        
        # 1. 优先尝试读取激光测距 (LRFTargetDistance)
        lrf_dist = xmp_data.get('LRFTargetDistance', '0')
        try:
            val = float(lrf_dist)
            if val > 0:
                distance_mm = val * 1000
                # print(f"Using LRF Distance: {val}m")
        except ValueError:
            pass

        # 2. 如果激光无效，尝试使用相对高度 (RelativeAltitude)
        if distance_mm == 0:
            rel_alt = xmp_data.get('RelativeAltitude', '0')
            try:
                val = float(rel_alt)
                if val != 0:
                    distance_mm = abs(val) * 1000 # 取绝对值防止负高度
                    print(f"Using Relative Altitude: {val}m")
            except ValueError:
                pass

        # 3. 如果还是 0 (说明是PNG或数据丢失)，才使用默认值兜底
        if distance_mm == 0:
            # 这里使用 config.py 里定义的默认距离
            distance_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000 
            print("Using Default Distance Estimate")

        # 收集第一张图的无人机信息
        if not self.global_drone_info and xmp_data:
            self.global_drone_info = {
                'Model': xmp_data.get('DroneModel', 'Unknown'),
                'Camera': xmp_data.get('ImageSource', 'Unknown'),
                'Firmware': xmp_data.get('Version', 'Unknown')
            }

        # 计算 GSD
        dist_str = xmp_data.get('LRFTargetDistance', '0')
        if float(dist_str) == 0:
             dist_str = xmp_data.get('RelativeAltitude', '0')
        
        distance_mm = float(dist_str) * 1000 
        
        # 计算 GSD (mm/pixel)
        gsd = calculate_gsd(
            distance_mm=distance_mm,
            focal_length_mm=focal_length,
            sensor_width_mm=config.SENSOR_WIDTH_MM,
            image_width_pix=img_w
        )

        # 获取位置元数据 (XYZ, Floor等)
        img_meta = self._get_metadata(img_path)

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
        stem_name = Path(img_path).stem
        os.makedirs(self.vis_dir, exist_ok=True)
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)

        img = Image.open(img_path).convert('RGB')
        
        # 1. 可视化
        img_vis = img.copy()
        if len(detections) > 0:
            img_vis = draw_box(img_vis, detections, self.labels, self.colors)
        vis_path = os.path.join(self.vis_dir, stem_name + '.png')
        img_vis.save(vis_path)

        # 2. 获取该图片的元数据 (新增)
        img_meta = self._get_metadata(img_path)

        # 3. 裁剪并记录
        records = []
        crops = crop_box(img, detections)
        
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            score = float(bbox[1])
            box = bbox[2:]
            
            cat_name = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge(box)
            
            crop_filename = f"{stem_name}_{i}_{cls_id}.png"
            crop_path = os.path.join(crop_subdir, crop_filename)
            crops[i].save(crop_path)

            # 基础记录
            record = {
                'Path': img_path,
                'VisPath': vis_path,
                'CropPath': crop_path,
                'Category': cat_name.title(),
                'Level': level,
                'Score': score,
                'Bbox': str(list(box)),
                'Action': action_judge(level),
            }
            # 合并元数据 (新增字段)
            record.update(img_meta)
            
            records.append(record)
        
        return pd.DataFrame(records)

    def run(self, output_path, model_name="Generic-Model", style_id=0):
        # 初始化目录
        base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(base_dir, 'report_vis')
        self.crop_dir = os.path.join(base_dir, 'report_crop')

        print("--- Loading Data ---")
        raw_data = self.loader.load()
        
        print("--- Processing Images ---")
        all_results_dfs = []
        img_paths_list = []
        
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
    engine.run(OUTPUT_PATH, model_name="BDD-MODEL", style_id=2)