import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 引入原有模块
from core.engine import ReportEngine
from core.processor import ImageProcessor
from exporters.dedup_exporter import PDFExporterDedup
import config

# 引入 Processor 需要的工具函数 (用于重写 process 方法)
from utils.geo_utils import calculate_gsd, pixel_to_physical
from utils.visualization import draw_box, crop_box
from utils.metadata import safe_float
from utils.analysis import level_judge, action_judge

class DedupProcessor(ImageProcessor):
    """
    专门处理包含 Track ID 的 7列数据 [cls, x1, y1, x2, y2, conf, id]
    """
    def process(self, item):
        img_path = item['image_path']
        detections = item['detections'] # 这里是 (N, 7) 的 numpy array
        stem_name = Path(img_path).stem
        
        # 1. 基础图像处理
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)

        # 2. GSD 计算 (复用原有逻辑)
        specs, _ = self.meta_mgr.get_camera_specs(all_meta, stem_name)
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        
        dist_mm = safe_float(all_meta.get('LRFTargetDistance')) * 1000
        if dist_mm == 0:
            dist_mm = abs(safe_float(all_meta.get('RelativeAltitude'))) * 1000
        if dist_mm == 0:
            dist_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000
        
        gsd = calculate_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 3. 可视化与裁剪 (关键修复点)
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        
        # 【Fix】: 这里的 detections 是 (N, 7)，draw_box 只要前 6 列
        vis_detections = detections[:, :6] if len(detections) > 0 else []
        
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        # 【Fix】: crop_box 也只要前 6 列
        crops = crop_box(img, vis_detections)

        # 4. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            # bbox 是 [cls, conf, x1, y1, x2, y2, id]
            cls_id = int(bbox[0])
            
            # 【Fix】: level_judge(bbox[2:]) 会拿到 5 个值，导致解包错误
            # 我们只传 bbox[1:5] 即 [x1, y1, x2, y2]
            level = level_judge(bbox[1:5])
            
            w_cm = pixel_to_physical(bbox[4]-bbox[2], gsd)
            h_cm = pixel_to_physical(bbox[5]-bbox[3], gsd)
            
            crop_p = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops):
                crops[i].save(crop_p)

            res = {
                'Category': self.labels[cls_id].title() if cls_id < len(self.labels) else f"Class_{cls_id}",
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action_judge(level),
                'W_pix': int(bbox[4]-bbox[2]),
                'H_pix': int(bbox[5]-bbox[3]),
                'Area_pix': int((bbox[4]-bbox[2]) * (bbox[5]-bbox[3])),
                'W_cm': f"{w_cm:.1f}" if w_cm else "N/A",
                'H_cm': f"{h_cm:.1f}" if h_cm else "N/A",
                'Area_cm2': f"{(w_cm * h_cm):.1f}" if (w_cm and h_cm) else "N/A",
                'VisPath': vis_path, 
                'CropPath': crop_p, 
                'Path': img_path,
                # 额外保存 ID 以便 Engine 使用，虽然 Engine 也可以通过 index 获取，但这里存一下更稳健
                'TrackID_Raw': int(bbox[-1]) 
            }
            if self.metadata_provider:
                ext_meta = self.metadata_provider(img_path)
                if ext_meta: res.update(ext_meta)
            records.append(res)
            
        return pd.DataFrame(records)


class DedupReportEngine(ReportEngine):
    def __init__(self, loader, labels, project_info_path, group_info_path, views_csv_path):
        super().__init__(loader, labels)
        self.proj_meta = self._load_json(project_info_path) # key: img_name -> list of dets
        self.group_meta = self._load_json(group_info_path)  # key: ID_xxx -> stats
        self.views_map = self.init_views_map(views_csv_path)

    def init_views_map(self, views_csv_path):
        # 假设 CSV 格式: View, Elevation (例如: V30, North)
        df_views = pd.read_csv(views_csv_path, header=0, index_col=False)
        # 转为字典 { 'V30': 'North', ... }
        # 兼容可能的列名大小写
        cols = [c.lower() for c in df_views.columns]
        if 'view' in cols and 'direction' in cols:
            df_views.columns = cols
            views_map = dict(zip(df_views['view'], df_views['direction']))
        else:
            print(f"Warning: CSV format not supported: {views_csv_path}")
            views_map = None
        return views_map


    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"Warning: JSON not found {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _enrich_record(self, record, img_name, track_id):
        """
        利用 project_info.json 中的数据增强 Processor 生成的记录
        """
        found_in_json = False
        if img_name in self.proj_meta:
            for item in self.proj_meta[img_name]:
                # 匹配 ID
                if item.get('id') == track_id:
                    # 1. Floor
                    record['Floor'] = item.get('floor', 'N/A')
                    
                    # 2. Projection World 数据 (注意 yolo_dedup 的嵌套结构)
                    # 结构通常是: "projection_world": { "z (height_m)": ..., "h (obj_height_m)": ... }
                    proj = item.get('projection_world', {})
                    
                    # 尝试多种可能的键名 (兼容不同版本的 dedup 脚本)
                    z_val = proj.get('z (height_m)', proj.get('z', 0))
                    h_val = proj.get('h (obj_height_m)', proj.get('h', 0))
                    
                    record['World_Z'] = f"{float(z_val):.2f}"
                    record['H_real_m'] = f"{float(h_val):.2f}m"

                    w_cm = record.get('W_cm', 0)
                    w_cm = float(w_cm) if w_cm else 0

                    h_cm = record.get('H_cm', 0)
                    h_cm = float(h_cm) if h_cm else 0
                    record['Real_Size'] = f"H: {h_cm:.1f}cm\nW: {w_cm:.1f}cm"
                    return record
        if not found_in_json:
            record['Floor'] = 'N/A'
            record['World_Z'] = 'N/A'
            # 如果没匹配到，就用 Processor 自己的 GSD 估算值
            w_cm = record.get('W_cm_val', 0)
            record['Real_Size'] = f"W: {w_cm:.1f}cm"
        return record

    def run(self, output_path, view_name="V01", model_name="BDD-MODEL"):
        # 1. 初始化目录
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis_fuse') 
        self.crop_dir = os.path.join(self.base_dir, 'report_crop_fuse')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        # 2. 加载数据
        raw_data = self.loader.load()
        if not raw_data: return

        # 3. 初始化自定义 Processor (使用 config 中的颜色)
        # 【Fix】使用 DedupProcessor 替代 ImageProcessor
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir)
        
        all_records = []
        view_elevation = self.views_map.get(view_name, "Unknown")
        print(f"Processing View: {view_name} (Elevation: {view_elevation})")
        for item in tqdm(raw_data, desc="Processing Images"):
            # 调用自定义的 process，不会再报错了
            df_res = processor.process(item)
            
            if df_res.empty: continue
            
            records = df_res.to_dict('records')
            detections = item['detections'] 
            
            img_stem = Path(item['image_path']).stem

            for i, rec in enumerate(records):
                if i < len(detections):
                    track_id = int(detections[i][6]) # 第7列是 ID
                    rec['ID'] = track_id
                    
                    # 注入 Dedup 计算的楼层和坐标
                    self._enrich_record(rec, img_stem, track_id)
                
                all_records.append(rec)

        # 4. 数据聚合
        final_df = pd.DataFrame(all_records)
        
        # 统计信息
        unique_ids = final_df['ID'].nunique() if not final_df.empty else 0
        
        report_data = {
            'input': {
                'number': len(raw_data), 
                'shape': (0,0,0,0), 
                'type': f'View {view_name} / Direction {view_elevation}',
            },
            'output': {
                'model': model_name, 
                'defects': unique_ids, 
                'no defects': 0, 
                'defects sta': self._get_id_stats(final_df)
            },
            'records': [final_df] 
        }

        # 5. 导出
        exporter = PDFExporterDedup()
        exporter.export(report_data, output_path)
        
        print(f"Report Generated: {output_path}")

    def _get_id_stats(self, df):
        if df.empty: return {}
        unique_df = df.drop_duplicates(subset=['ID'])
        return unique_df['Category'].value_counts().to_dict()