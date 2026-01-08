import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 复用基础类
from core.engine import ReportEngine
from core.processor import ImageProcessor
import config
from utils.geo_utils import calculate_gsd, pixel_to_physical
from utils.visualization import draw_box, crop_box
from utils.metadata import safe_float
from utils.analysis import level_judge, action_judge
from exporters import EXPORTER_MAP
import re  # <--- 别忘了导入 re


def parse_dms_to_dd(dms_str):
    """
    解析 DMS 格式字符串 (e.g. '22 deg 18' 35.81" N') 为十进制度数
    """
    if not isinstance(dms_str, str):
        return 0.0
    
    # 匹配模式: 数字 deg 数字' 数字" 方向
    pattern = r"(\d+)\s*deg\s*(\d+)'\s*([\d.]+)\"\s*([NSEW])"
    match = re.search(pattern, dms_str)
    
    if match:
        deg = float(match.group(1))
        minute = float(match.group(2))
        second = float(match.group(3))
        direction = match.group(4)
        
        # 核心公式: 度 + 分/60 + 秒/3600
        dd = deg + minute/60 + second/3600
        
        # 南纬(S)和西经(W)为负数
        if direction in ['S', 'W']:
            dd = -dd
            
        return dd
    
    # 兜底：如果格式不对，尝试直接转换 (防止已经是数字的情况)
    try:
        return float(dms_str)
    except:
        return 0.0



class DedupProcessor(ImageProcessor):
    """
    专用处理器：负责图像处理 + 提取原始 GPS 信息
    """
    def process(self, item):
        img_path = item['image_path']
        detections = item['detections']
        stem_name = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        
        # 1. 提取元数据 (包含 GPS)
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)
        specs, _ = self.meta_mgr.get_camera_specs(all_meta, stem_name)
        
        # 提取 GPS 信息
        lat = parse_dms_to_dd(all_meta.get('GPSLatitude'))
        lon = parse_dms_to_dd(all_meta.get('GPSLongitude'))
        alt = safe_float(all_meta.get('AbsoluteAltitude')) or safe_float(all_meta.get('GPSAltitude'))
        
        # 格式化 GPS 字符串
        gps_str = "N/A"
        if lat != 0.0 and lon != 0.0:
            # 判断纬度方向
            lat_dir = "N" if lat >= 0 else "S"
            # 判断经度方向
            lon_dir = "E" if lon >= 0 else "W"
            
            # 取绝对值显示，并加上字母
            gps_str = f"{abs(lat):.6f}{lat_dir}, {abs(lon):.6f}{lon_dir}"
            
            if alt:
                gps_str += f"\nAlt: {alt:.1f}m"
        
        # GSD 计算
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        dist_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000 
        gsd = calculate_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 2. 可视化
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        vis_detections = detections[:, :] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)

        # 3. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            track_id = int(bbox[6]) # Dedup ID
            
            level = level_judge(bbox[2:6])
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)
            
            crop_p = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_p)

            res = {
                'Path': img_path,
                'VisPath': vis_path, 
                'CropPath': crop_p, 
                'Category': self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}",
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action_judge(level),
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                'W_cm': float(f"{w_cm:.2f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.2f}") if h_cm else "N/A", 
                'Area_cm2': "N/A",
                
                # === 注入 GPS ===
                'GPS': gps_str,
                
                # 内部字段
                '_track_id': track_id,
                '_stem_name': stem_name
            }
            records.append(res)
            
        return pd.DataFrame(records)

class DedupReportEngine(ReportEngine):
    """
    Dedup 专用引擎
    """
    def __init__(self, loader, labels, project_info_path, group_info_path, views_csv_path=None, target_class_names=None, floor_map_path=None):
        super().__init__(loader, labels)
        if hasattr(loader, 'target_class_names') and loader.target_class_names:
            self.labels = loader.target_class_names
        elif target_class_names:
            self.labels = target_class_names
        else:
            self.labels = None
            print("[WARN] No target class names provided, using default labels.")

        self.proj_meta = self._load_json(project_info_path)
        self.views_map = self._load_views_map(views_csv_path)

        # === 【新增】只读取 floor_map 的 keys 作为定义的楼层列表 ===
        floor_config = self._load_json(floor_map_path) if floor_map_path else {}
        self.defined_floors = list(floor_config.keys()) if floor_config else []
    def _load_json(self, path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)

    def _load_views_map(self, path):
        if not path or not os.path.exists(path): return {}
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]
            if 'view' in cols and 'direction' in cols:
                df.columns = cols
                return dict(zip(df['view'], df['direction']))
        except: return {}
        return {}

    def _enrich_data(self, df, view_name):
        """注入物理信息 (Floor, Real H)"""
        if df.empty: return df
        
        floors, ids, orientations = [], [], []
        
        # 获取朝向
        ele_str = self.views_map.get(view_name.strip(), self.views_map.get(view_name, "Unknown"))

        for idx, row in df.iterrows():
            track_id = row['_track_id']
            img_name = row['_stem_name']
            
            fl = "N/A"
            
            # 从 project_info.json 匹配 Dedup 结果
            if img_name in self.proj_meta:
                for item in self.proj_meta[img_name]:
                    if item.get('id') == track_id:
                        fl = item.get('floor', 'N/A')
                        
                        # 使用真实高度覆盖 GSD 高度
                        proj = item.get('projection_world', {})
                        h_real_m = proj.get('h (obj_height_m)', proj.get('h', 0))
                        if h_real_m > 0:
                            real_h_cm = float(h_real_m) * 100
                            df.at[idx, 'H_cm'] = real_h_cm
                            w_cm = row['W_cm']
                            if isinstance(w_cm, float):
                                df.at[idx, 'Area_cm2'] = float(f"{w_cm * real_h_cm:.1f}")
                        break
            
            floors.append(fl)
            ids.append(track_id)
            orientations.append(ele_str)

        df['ID'] = ids
        df['floor'] = floors
        df['view'] = view_name
        df['orientation'] = orientations
        return df

    def run(self, output_path, view_name="V01", model_name="BDD-MODEL", style_id=3):
        # 1. 目录初始化
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis_fuse') 
        self.crop_dir = os.path.join(self.base_dir, 'report_crop_fuse')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        raw_data = self.loader.load()
        if not raw_data: return

        # 2. 处理图像
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir)
        all_dfs = []
        
        print(f"Processing View: {view_name}...")
        for item in tqdm(raw_data, desc="Processing Images"):
            df = processor.process(item)
            if not df.empty:
                df = self._enrich_data(df, view_name)
                all_dfs.append(df)

        if not all_dfs:
            print("No defects found.")
            return

        # 3. 组织数据
        # Style 3 (Dedup): 聚合为一个大表
        final_records = []
        if style_id == 3:
            print("Organizing data for Compact Report (Merged View)...")
            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_df = merged_df.sort_values(by=['ID', 'floor'])
            final_records = [merged_df] # 只有一个大表
        else:
            final_records = all_dfs

        # 4. 统计信息
        full_df = pd.concat(all_dfs, ignore_index=True)
        unique_ids = full_df['ID'].nunique() if not full_df.empty else 0
        
        report_data = {
            'input': {
                'number': len(raw_data), 
                'shape': (0,0,0,0), 
                'type': f'{view_name}'
            },
            'output': {
                'model': model_name, 
                'defects': unique_ids, 
                'no defects': 0, 
                'defects sta': full_df.drop_duplicates(subset=['ID'])['Category'].value_counts().to_dict(),
                'elevation': self.views_map.get(view_name, '')
            },
            'records': final_records,
            'defined_categories': self.labels,
            'defined_floors': self.defined_floors
        }

        # 5. 导出
        ExporterClass = EXPORTER_MAP.get(style_id)
        if not ExporterClass: return
            
        exporter = ExporterClass()
        exporter.export(report_data, output_path)
        
        print(f"Report Generated: {output_path}")
    

