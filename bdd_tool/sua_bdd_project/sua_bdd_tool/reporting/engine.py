# core/processor.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time

from PIL import Image
import pandas as pd
from tqdm import tqdm

import config
from sua_bdd_tool.data.image_meta import MetadataManager
from sua_bdd_tool.data.loaders import DedupLoader, DedupLoader_AuxImage
from sua_bdd_tool.utils.analysis import action_judge, img_sta, level_judge
from sua_bdd_tool.utils.projection import calculate_facade_gsd, pixel_to_physical, safe_float, convert_coordinate
from sua_bdd_tool.utils.visualization import crop_box, draw_box

from . import EXPORTER_MAP

class ImageProcessor:
    def __init__(self, labels, colors, vis_dir, crop_dir, vis_aux_dir=None, crop_aux_dir=None, metadata_provider=None, exif_db=None):
        self.meta_mgr = MetadataManager()
        self.labels = labels
        self.colors = colors
        self.vis_dir = vis_dir
        self.vis_aux_dir = vis_aux_dir
        self.crop_dir = crop_dir
        self.crop_aux_dir = crop_aux_dir
        self.metadata_provider = metadata_provider
        self.exif_db = exif_db

    def process(self, item):
        img_path = item['image_path']
        detections = item['detections']
        stem_name = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)

        # 硬件与GSD计算
        specs, _ = self.meta_mgr.get_camera_specs(all_meta, stem_name)
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        
        dist_mm = safe_float(all_meta.get('LRFTargetDistance')) * 1000
        if dist_mm == 0:
            dist_mm = abs(safe_float(all_meta.get('RelativeAltitude'))) * 1000
        if dist_mm == 0:
            import config
            dist_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000
        
        gsd = calculate_facade_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 可视化
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        draw_box(img.copy(), detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, detections)

        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            w_cm = pixel_to_physical(bbox[4]-bbox[2], gsd)
            h_cm = pixel_to_physical(bbox[5]-bbox[3], gsd)


            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)

            crop_p = os.path.join(crop_subdir, f"{i}.png")
            crops[i].save(crop_p)

            res = {
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
                'W_pix': int(bbox[4]-bbox[2]),
                'H_pix': int(bbox[5]-bbox[3]),
                'Area_pix': int((bbox[4]-bbox[2]) * (bbox[5]-bbox[3])),
                'W_cm': f"{w_cm:.1f}" if w_cm else "N/A",
                'H_cm': f"{h_cm:.1f}" if h_cm else "N/A",
                'Area_cm2': f"{(w_cm * h_cm):.1f}" if (w_cm and h_cm) else "N/A",
                'VisPath': vis_path, 'CropPath': crop_p, 'Path': img_path
            }
            if self.metadata_provider:
                ext_meta = self.metadata_provider(img_path)
                if ext_meta: res.update(ext_meta)
            records.append(res)
            
        return pd.DataFrame(records)


class ReportEngine:
    def __init__(self, loader, labels, metadata_getter=None):
        self.loader = loader
        self.labels = labels
        self.metadata_getter = metadata_getter
        self.meta_mgr = MetadataManager()
        self._init_metadata_store()

    def _init_metadata_store(self):
        self.global_drone_info = {}
        self.base_dir = None
        self.vis_dir = None
        self.crop_dir = None

    def _declare_metadata(self, img_path):
        img = Image.open(img_path)
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)
        specs, model = self.meta_mgr.get_camera_specs(all_meta, Path(img_path).stem)
        
        self.global_drone_info = {
            'Model': model,
            'Camera': all_meta.get('ImageSource', 'Unknown'),
            'Firmware': all_meta.get('Firmware', all_meta.get('Version', 'Unknown'))
        }
        
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        
        print("\n" + "="*50)
        print(f"--- [Metadata Declaration] ---")
        print(f" Source Image   : {Path(img_path).name}")
        print(f" Parsing Method : {all_meta.get('_parsing_method')}")
        print("-" * 50)
        print(f" Drone Model    : {model}")
        print(f" Sensor Width   : {specs['sensor_width_mm']} mm")
        print(f" Focal Length   : {focal} mm")
        print("="*50 + "\n")
        
    def _prepare_report_data(self, all_dfs, img_paths, model_name):
        """
        聚合所有 DataFrame 结果并生成报告摘要信息
        """
        # 过滤掉可能的空结果（虽然 process 应该总返回 DF，但为了稳健性）
        valid_dfs = [df for df in all_dfs if df is not None]
        
        # 计算检测到缺陷的图像数量
        has_defect_count = sum(1 for df in valid_dfs if not df.empty)
        
        # 统计各类别缺陷总数
        cat_counts = {}
        for df in valid_dfs:
            if not df.empty:
                counts = df['Category'].value_counts()
                for cat, count in counts.items():
                    cat_counts[cat] = cat_counts.get(cat, 0) + count

        return {
            'input': {
                'number': len(img_paths), 
                'shape': img_sta(img_paths), # 统计图片最大最小尺寸
                'type': 'Images'
            },
            'output': {
                'model': model_name, 
                'defects': has_defect_count, 
                'no defects': len(img_paths) - has_defect_count, 
                'defects sta': cat_counts
            },
            'records': valid_dfs,
            'drone_info': self.global_drone_info
        }

    def run(self, output_path, model_name="BDD-MODEL", style_id=3, use_multithreading=True, max_workers=4):
        """
        :param use_multithreading: 是否启用多线程加速
        :param max_workers: 线程个数
        """
        # 1. 目录初始化与数据加载
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis')
        self.crop_dir = os.path.join(self.base_dir, 'report_crop')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        raw_data = self.loader.load()
        if not raw_data: return

        # 2. 环境声明
        self._declare_metadata(raw_data[0]['image_path'])

        # 3. 图像解析逻辑
        processor = ImageProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir, self.metadata_getter)
        all_dfs = [None] * len(raw_data)
        img_paths = [item['image_path'] for item in raw_data]

        if use_multithreading:
            # --- 多线程模式 ---
            print(f"[{time.strftime('%H:%M:%S')}] Starting multi-threaded processing ({max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {executor.submit(processor.process, item): i for i, item in enumerate(raw_data)}
                for future in tqdm(as_completed(future_to_index), total=len(raw_data), desc="Processing"):
                    all_dfs[future_to_index[future]] = future.result()
        else:
            # --- 单线程模式 ---
            print(f"[{time.strftime('%H:%M:%S')}] Starting single-threaded processing...")
            for i, item in enumerate(tqdm(raw_data, desc="Processing")):
                all_dfs[i] = processor.process(item)

        # 4. 统计与导出 (保持不变)
        report_info = self._prepare_report_data(all_dfs, img_paths, model_name)
        ExporterClass = EXPORTER_MAP.get(style_id, EXPORTER_MAP[0])
        ExporterClass().export(report_info, output_path)
        
        self._init_metadata_store()
        print("--- Run Completed ---")


class DedupProcessor(ImageProcessor):
    """
    专用处理器：负责图像处理 + 提取原始 GPS 信息
    """
    def process(self, item):
        img_path = item['image_path']
        detections = item['detections']
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')

        img_exif = self.exif_db.get(img_name, None)
        focal_length = img_exif.get('focal', None) if img_exif else None
        lrf_dist = img_exif.get('lrf_dist', None) if img_exif else None
        lon = img_exif.get('lon', None) if img_exif else None   
        lat = img_exif.get('lat', None) if img_exif else None   
        gps_str = convert_coordinate(lat, lon)

        if img_exif["model"] in ["DJI M4T", "M4T"]:
            if img_exif["camera_type"] in ["WideCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Wide"]["pixel_size_um"]
            elif img_exif["camera_type"] in ["InfraredCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Thermal"]["pixel_size_um"]
            else:
                raise ValueError(f"Unknown camera_type: {img_exif['camera_type']}")
        
        gsd = calculate_facade_gsd(lrf_dist, focal_length, pixel_size_um)


        # 2. 可视化
        vis_path = os.path.join(self.vis_dir, f"{img_stem}.png")
        vis_detections = detections[:, :] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, img_stem)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)

        # 3. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            id = int(bbox[6]) # Dedup ID
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)

            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)
            
            crop_path = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_path)

            res = {
                'Path': img_path,
                'VisPath': vis_path, 
                'CropPath': crop_path, 
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                'W_cm': float(f"{w_cm:.2f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.2f}") if h_cm else "N/A", 
                'Area_cm2': w_cm*h_cm if (w_cm and h_cm) else "N/A",
                
                # === 注入 GPS ===
                'GPS': gps_str,
                
                # 内部字段
                'id': id,
                'img_name': img_name
            }
            records.append(res)
            
        return pd.DataFrame(records)


class DedupProcessor_AuxImage(ImageProcessor):
    """
    专用处理器：负责图像处理 + 提取原始 GPS 信息
    """
    def process(self, item):
        detections = item['detections']
        img_path = item['image_path']
        img_aux_path = item['image_aux_path']
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem
        # img_aux_name = Path(img_aux_path).name
        # img_aux_stem = Path(img_aux_path).stem

        img = Image.open(img_path).convert('RGB')
        img_aux = Image.open(img_aux_path).convert('RGB')

        img_exif = self.exif_db.get(img_name, None)
        focal_length = img_exif.get('focal', None) if img_exif else None
        lrf_dist = img_exif.get('lrf_dist', None) if img_exif else None
        lon = img_exif.get('lon', None) if img_exif else None   
        lat = img_exif.get('lat', None) if img_exif else None   
        gps_str = convert_coordinate(lat, lon)

        if img_exif["model"] in ["DJI M4T", "M4T"]:
            if img_exif["camera_type"] in ["WideCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Wide"]["pixel_size_um"]
            elif img_exif["camera_type"] in ["InfraredCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Thermal"]["pixel_size_um"]
            else:
                raise ValueError(f"Unknown camera_type: {img_exif['camera_type']}")
        
        gsd = calculate_facade_gsd(lrf_dist, focal_length, pixel_size_um)


        # 2. 可视化
        vis_path = os.path.join(self.vis_dir, f"{img_stem}.png")
        vis_aux_path = os.path.join(self.vis_aux_dir, f"{img_stem}.png")

        vis_detections = detections[:, :] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        draw_box(img_aux.copy(), vis_detections, self.labels, self.colors).save(vis_aux_path)
        
        crop_subdir = os.path.join(self.crop_dir, img_stem)
        crop_aux_subdir = os.path.join(self.crop_aux_dir, img_stem)
        os.makedirs(crop_subdir, exist_ok=True)
        os.makedirs(crop_aux_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)
        crops_aux = crop_box(img_aux, vis_detections)

        # 3. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            id = int(bbox[6]) # Dedup ID
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)

            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)
            
            crop_path = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_path)
            crop_aux_path = os.path.join(crop_aux_subdir, f"{i}.png")
            if i < len(crops_aux): crops_aux[i].save(crop_aux_path)
            
            res = {
                'Path': img_path,
                'VisPath': vis_path, 
                'VisAuxPath': vis_aux_path,
                'CropPath': crop_path, 
                'CropAuxPath': crop_aux_path,
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                'W_cm': float(f"{w_cm:.2f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.2f}") if h_cm else "N/A", 
                'Area_cm2': w_cm*h_cm if (w_cm and h_cm) else "N/A",
                
                # === 注入 GPS ===
                'GPS': gps_str,
                
                # 内部字段
                'id': id,
                'img_name': img_name
            }
            records.append(res)
            
        return pd.DataFrame(records)



class DedupReportEngine(ReportEngine):
    """
    Dedup 专用引擎 (支持多线程)
    """
    def __init__(self, loader, labels, project_info_path, group_info_path, views_csv_path=None, target_class_names=None, floor_map_path=None, exif_db=None):
        super().__init__(loader, labels)
        self.exif_db=exif_db
        if hasattr(loader, 'target_class_names') and loader.target_class_names:
            self.labels = loader.target_class_names
        elif target_class_names:
            self.labels = target_class_names
        else:
            self.labels = None
            print("[WARN] No target class names provided, using default labels.")

        self.proj_meta = self._load_json(project_info_path)
        self.views_map = self._load_views_map(views_csv_path)

        floor_config = self._load_json(floor_map_path)['floor_map'] if floor_map_path else {}
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
        """注入物理信息 (Floor, Real H) - 逻辑保持不变"""
        if df.empty: return df
        
        floors, ids, orientations = [], [], []
        ele_str = self.views_map.get(view_name.strip(), self.views_map.get(view_name, "Unknown"))

        for idx, row in df.iterrows():
            id = row['id']
            img_name = row['img_name']
            
            fl = "N/A"
            if img_name in self.proj_meta:
                for item in self.proj_meta[img_name]:
                    if item.get('id') == id:
                        fl = item.get('floor', 'N/A')
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
            ids.append(id)
            orientations.append(ele_str)

        df['ID'] = ids
        df['floor'] = floors
        df['view'] = view_name
        df['orientation'] = orientations
        return df

    def run(self, output_path, view_name="V01", model_name="BDD-MODEL", style_id=3, use_multithreading=True, max_workers=4):
        """
        :param use_multithreading: 是否启用多线程加速 [新增]
        :param max_workers: 线程个数 [新增]
        """
        # 1. 目录初始化
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis_fuse') 
        self.crop_dir = os.path.join(self.base_dir, 'report_crop_fuse')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        raw_data = self.loader.load()
        if not raw_data: return

        # 2. 处理图像 (多线程改造部分)
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir)
        
        # 预分配列表以保持顺序 (raw_results 存储 processor.process 的原始返回)
        raw_results = [None] * len(raw_data)
        
        print(f"Processing View: {view_name}...")

        if use_multithreading:
            # --- 多线程模式 ---
            print(f"[{time.strftime('%H:%M:%S')}] Starting multi-threaded processing ({max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务：只执行 heavy lifting 的 processor.process
                future_to_index = {executor.submit(processor.process, item): i for i, item in enumerate(raw_data)}
                
                for future in tqdm(as_completed(future_to_index), total=len(raw_data), desc="Processing Images"):
                    idx = future_to_index[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as e:
                        print(f"❌ Error processing image index {idx}: {e}")
        else:
            # --- 单线程模式 ---
            print(f"[{time.strftime('%H:%M:%S')}] Starting single-threaded processing...")
            for i, item in enumerate(tqdm(raw_data, desc="Processing Images")):
                raw_results[i] = processor.process(item)

        # 3. 后处理：过滤空结果并注入元数据 (Enrich Data)
        # 注意：_enrich_data 很快且涉及类成员读取，在主线程串行执行更安全且不影响性能
        all_dfs = []
        for df in raw_results:
            if df is not None and not df.empty:
                # 这一步将 GPS、楼层等信息注入 DataFrame
                enriched_df = self._enrich_data(df, view_name)
                all_dfs.append(enriched_df)

        if not all_dfs:
            print("No defects found.")
            return

        # 4. 组织数据 (保持不变)
        final_records = []
        if style_id == 3:
            print("Organizing data for Compact Report (Merged View)...")
            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_df = merged_df.sort_values(by=['ID', 'floor'])
            final_records = [merged_df] 
        else:
            final_records = all_dfs

        # 5. 统计信息 (保持不变)
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

        # 6. 导出
        ExporterClass = EXPORTER_MAP.get(style_id)
        if not ExporterClass: return
            
        exporter = ExporterClass()
        exporter.export(report_data, output_path)
        
        print(f"Report Generated: {output_path}")


class BatchDedupEngine(DedupReportEngine):
    """
    派生类：用于批量处理 View 并生成汇总报告。
    """
    
    def __init__(self, *args, **kwargs):
        # 允许初始化时不传 loader，后续动态加载
        if 'loader' not in kwargs:
            kwargs['loader'] = None 
        super().__init__(*args, **kwargs)

    def process_view_data(self, view_id, img_dir, label_dir, project_info_path, class_path, target_cls_ids=None):
        """
        [核心扩展方法]
        处理单个 View，返回 DataFrame，但不生成 PDF。
        """
        print(f"--- [Batch] Collecting data for {view_id} ---")
        
        # 1. 动态实例化 Loader
        current_loader = DedupLoader(
            img_dir=img_dir, 
            txt_dir=label_dir, 
            class_path=class_path, 
            target_cls_ids=target_cls_ids,
            exif_db=self.exif_db,
        )
        
        # ================== 【修复点开始】 ==================
        # 关键修复：如果当前 Engine 没有标签（None或空），从 Loader 中获取
        # DedupLoader 初始化时会自动读取 config.CLASS_PATH 并生成 target_class_names
        if not self.labels and hasattr(current_loader, 'target_class_names'):
            if current_loader.target_class_names:
                self.labels = current_loader.target_class_names
                # print(f"    [Info] Labels updated from loader: {self.labels}")
        # ================== 【修复点结束】 ==================
        
        # 2. 更新项目元数据
        if os.path.exists(project_info_path):
             self.proj_meta = self._load_json(project_info_path)
        else:
             self.proj_meta = {}

        # 3. 加载原始数据
        raw_data = current_loader.load()
        if not raw_data: 
            print(f"    No data found for {view_id}")
            return pd.DataFrame()

        # 4. 准备图片输出路径
        if not hasattr(self, 'vis_dir') or self.vis_dir is None:
            self.vis_dir = os.path.join(label_dir, '../batch_vis')
            self.crop_dir = os.path.join(label_dir, '../batch_crop')
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.crop_dir, exist_ok=True)

        # 5. 处理图像 (确保传入了修复后的 self.labels)
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir, exif_db=self.exif_db)
        
        view_dfs = []
        for item in tqdm(raw_data, desc=f"    Analyzing {view_id}", leave=False):
            df = processor.process(item)
            if not df.empty:
                df = self._enrich_data(df, view_id)
                view_dfs.append(df)
        
        if not view_dfs:
            return pd.DataFrame()
            
        return pd.concat(view_dfs, ignore_index=True)

    def process_view_data_aux(self, view_id, img_dir, img_aux_dir, label_dir, project_info_path, class_path, target_cls_ids=None):
        """
        [核心扩展方法]
        处理单个 View，返回 DataFrame，但不生成 PDF。
        """
        print(f"--- [Batch] Collecting data for {view_id} ---")
        
        # 1. 动态实例化 Loader
        current_loader = DedupLoader_AuxImage(
            img_dir=img_dir, 
            img_aux_dir=img_aux_dir,
            txt_dir=label_dir, 
            class_path=class_path, 
            target_cls_ids=target_cls_ids,
            exif_db=self.exif_db,
        )
        
        # ================== 【修复点开始】 ==================
        # 关键修复：如果当前 Engine 没有标签（None或空），从 Loader 中获取
        # DedupLoader 初始化时会自动读取 config.CLASS_PATH 并生成 target_class_names
        if not self.labels and hasattr(current_loader, 'target_class_names'):
            if current_loader.target_class_names:
                self.labels = current_loader.target_class_names
                # print(f"    [Info] Labels updated from loader: {self.labels}")
        # ================== 【修复点结束】 ==================
        
        # 2. 更新项目元数据
        if os.path.exists(project_info_path):
             self.proj_meta = self._load_json(project_info_path)
        else:
             self.proj_meta = {}

        # 3. 加载原始数据
        raw_data = current_loader.load()
        if not raw_data: 
            print(f"    No data found for {view_id}")
            return pd.DataFrame()

        # 4. 准备图片输出路径
        if not hasattr(self, 'vis_dir') or self.vis_dir is None:
            self.vis_dir = os.path.join(label_dir, '../batch_vis')
            self.crop_dir = os.path.join(label_dir, '../batch_crop')
            self.vis_aux_dir = os.path.join(label_dir, '../batch_vis_aux')
            self.crop_aux_dir = os.path.join(label_dir, '../batch_crop_aux')
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.crop_dir, exist_ok=True)
            os.makedirs(self.vis_aux_dir, exist_ok=True)
            os.makedirs(self.crop_aux_dir, exist_ok=True)

        # 5. 处理图像 (确保传入了修复后的 self.labels)
        processor = DedupProcessor_AuxImage(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir, self.vis_aux_dir, self.crop_aux_dir, exif_db=self.exif_db)
        
        view_dfs = []
        for item in tqdm(raw_data, desc=f"Analyzing {view_id}", leave=False):
            df = processor.process(item)
            if not df.empty:
                df = self._enrich_data(df, view_id)
                view_dfs.append(df)
        
        if not view_dfs:
            return pd.DataFrame()
            
        return pd.concat(view_dfs, ignore_index=True)

    def export_aggregated_report(self, all_df, output_path, model_name="BDD-MODEL", style_id=4):
        """
        [核心扩展方法]
        接收一个包含所有 View 数据的大 DataFrame，并生成 PDF。
        """
        if all_df.empty:
            print("[ERROR] No aggregated data to export.")
            return

        print(f">>> [Batch] Generating Aggregated Report: {output_path}")
        
        # 1. 智能排序
        if 'view' in all_df.columns:
            all_df['view_num'] = all_df['view'].apply(
                lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else 999
            )
            all_df = all_df.sort_values(by=['view_num', 'ID'])
            all_df = all_df.drop(columns=['view_num'])

        # 2. 构建 Report Data 字典
        unique_ids = all_df['ID'].nunique()
        view_list = sorted(all_df['view'].unique().astype(str))
        view_range_str = f"{view_list[0]}~{view_list[-1]}" if len(view_list) > 1 else view_list[0]

        report_data = {
            'input': {
                'number': all_df['Path'].nunique(), 
                'shape': (0,0,0,0), 
                'type': f'Aggregated Views ({view_range_str})'
            },
            'output': {
                'model': model_name, 
                'defects': unique_ids, 
                'no defects': 0, 
                'defects sta': all_df.drop_duplicates(subset=['ID'])['Category'].value_counts().to_dict(),
                'elevation': "All Directions"
            },
            'records': [all_df], 
            'defined_categories': self.labels,
            'defined_floors': self.defined_floors
        }

        # 3. 调用 Exporter
        ExporterClass = EXPORTER_MAP.get(style_id)
        if not ExporterClass: 
            print(f"Style {style_id} not found.")
            return
            
        exporter = ExporterClass()
        exporter.export(report_data, output_path)
        print(f"Done! Saved to {output_path}")