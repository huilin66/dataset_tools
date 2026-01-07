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

class DedupProcessor(ImageProcessor):
    """
    专用处理器：负责图像层面的处理 (画框、GSD、裁剪)，
    但返回的数据结构完全兼容标准格式。
    """
    def process(self, item):
        img_path = item['image_path']
        detections = item['detections'] # 7列数据
        stem_name = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        
        # 1. 基础 GSD 计算 (用于计算 Width，因为 Dedup 只提供 Height)
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)
        specs, _ = self.meta_mgr.get_camera_specs(all_meta, stem_name)
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        dist_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000 
        gsd = calculate_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 2. 可视化 (前6列)
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        vis_detections = detections[:, :6] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)

        # 3. 生成记录 (兼容标准字段 + 预留 Dedup 字段)
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            track_id = int(bbox[6]) # Dedup ID
            
            level = level_judge(bbox[2:6])
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd) # 这里的 H_cm 稍后会被 Dedup 的真实高度覆盖
            
            crop_p = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_p)

            res = {
                # --- 标准字段 (所有 Exporter 都认) ---
                'Path': img_path,
                'VisPath': vis_path, 
                'CropPath': crop_p, 
                'Category': self.labels[cls_id].title() if cls_id < len(self.labels) else f"Class_{cls_id}",
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action_judge(level),
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                
                # 初始物理尺寸 (暂存 GSD 结果)
                'W_cm': float(f"{w_cm:.1f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.1f}") if h_cm else "N/A", 
                'Area_cm2': "N/A", # 稍后重算
                
                # --- 内部字段 (用于 Engine 注入数据) ---
                '_track_id': track_id,
                '_stem_name': stem_name
            }
            records.append(res)
            
        return pd.DataFrame(records)

class DedupReportEngine(ReportEngine):
    """
    Dedup 专用引擎：
    1. 整合了原 dedup_exporter 的数据组织逻辑。
    2. 负责读取 JSON 并注入物理信息。
    3. 根据 style_id 决定输出格式（单图列表 vs 聚合大表）。
    """
    def __init__(self, loader, labels, project_info_path, group_info_path, views_csv_path=None):
        super().__init__(loader, labels)
        self.proj_meta = self._load_json(project_info_path)
        self.views_map = self._load_views_map(views_csv_path)

    def _load_json(self, path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)

    def _load_views_map(self, path):
        if not path or not os.path.exists(path): return {}
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]
            if 'view' in cols and 'elevation' in cols:
                df.columns = cols
                return dict(zip(df['view'], df['elevation']))
        except: return {}
        return {}

    def _enrich_data(self, df, view_name):
        """注入物理信息 (Floor, XYZ, Real H) 并重命名 ID 列"""
        if df.empty: return df
        
        floors, xyzs, ids, orientations = [], [], [], []
        
        ele_str = self.views_map.get(view_name.strip(), self.views_map.get(view_name, "Unknown"))

        for idx, row in df.iterrows():
            track_id = row['_track_id']
            img_name = row['_stem_name']
            
            fl, z_str, x_str = "N/A", "N/A", "N/A"
            
            # 从 project_info.json 匹配
            if img_name in self.proj_meta:
                for item in self.proj_meta[img_name]:
                    if item.get('id') == track_id:
                        fl = item.get('floor', 'N/A')
                        
                        proj = item.get('projection_world', {})
                        z_val = proj.get('z (height_m)', proj.get('z', 0))
                        x_val = proj.get('x (horizontal_m)', proj.get('x', 0))
                        z_str = f"{float(z_val):.1f}m"
                        x_str = f"{float(x_val):.1f}m"
                        
                        # 使用真实高度覆盖 GSD 高度
                        h_real_m = proj.get('h (obj_height_m)', proj.get('h', 0))
                        if h_real_m > 0:
                            real_h_cm = float(h_real_m) * 100
                            df.at[idx, 'H_cm'] = real_h_cm
                            # 更新面积 (W_cm * Real_H_cm)
                            w_cm = row['W_cm']
                            if isinstance(w_cm, float):
                                df.at[idx, 'Area_cm2'] = float(f"{w_cm * real_h_cm:.1f}")
                        break
            
            floors.append(fl)
            xyzs.append(f"Z:{z_str}, X:{x_str}")
            ids.append(track_id)
            orientations.append(ele_str)

        # 写入兼容字段
        df['ID'] = ids          # 核心：标准 Exporter 会优先读这个
        df['floor'] = floors    # 核心：Detailed/Compact 模式会读这个
        df['xyz'] = xyzs        # 核心：Detailed/Compact 模式会读这个
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
                # 立即注入数据，保证每张图的 DataFrame 都是完整的
                df = self._enrich_data(df, view_name)
                all_dfs.append(df)

        if not all_dfs:
            print("No defects found.")
            return

        # 3. [核心逻辑] 根据 Style 决定数据组织方式
        # Style 3 (Compact) 本质是 Excel 清单，用户期望看到 ID 排序的聚合大表
        # Style 0/1/2 (Basic/Detailed) 是按页展示的，必须保持按图片分页
        
        final_records = []
        
        if style_id == 3:
            # === 聚合模式 ===
            print("Organizing data for Compact Report (Merged View)...")
            merged_df = pd.concat(all_dfs, ignore_index=True)
            # 按 ID 排序 (复现 dedup_exporter 的逻辑)
            merged_df = merged_df.sort_values(by=['ID', 'floor'])
            final_records = [merged_df] # 只有一个大表
        else:
            # === 标准模式 (保持原样) ===
            print(f"Organizing data for Style {style_id} (Per Image)...")
            final_records = all_dfs

        # 4. 统计信息准备
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
            'records': final_records # 传入组织好的数据列表
        }

        # 5. 调用 Exporter (直接使用 pdf_styles 中的标准 Exporter)
        ExporterClass = EXPORTER_MAP.get(style_id)
        if not ExporterClass:
            print(f"Error: Style {style_id} not found.")
            return
            
        exporter = ExporterClass()
        exporter.export(report_data, output_path)
        
        print(f"Report Generated: {output_path}")