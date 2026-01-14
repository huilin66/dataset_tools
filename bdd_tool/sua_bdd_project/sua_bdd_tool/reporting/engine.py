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
from sua_bdd_tool.utils.analysis import img_sta
from sua_bdd_tool.utils.projection import safe_float

from . import EXPORTER_MAP
from sua_bdd_tool.data.processor import ImageProcessor, DedupProcessor, DedupProcessor_AuxImage

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


class DedupReportEngine(ReportEngine):
    """
    Dedup 专用引擎 (支持多线程)
    """
    def __init__(self, loader=None, labels=None, project_info_path=None, group_info_path=None, 
                 views_csv_path=None, target_class_names=None, floor_map_path=None, exif_db=None):
        
        # 调用父类初始化，允许 loader 为 None
        super().__init__(loader, labels)
        self.exif_db = exif_db
        
        # 2. Label 解析逻辑 (保持上一轮优化后的逻辑)
        self.labels = self._resolve_labels(loader, target_class_names)
        # 如果初始化时既没给 label 也没给 loader，这里暂时允许为空，留给后续动态加载
        if not self.labels and loader: 
             print("[WARN] No target class names provided, using default labels.")

        # 3. 资源加载 (增加 None 判断，避免报错)
        self.project_info_path = project_info_path
        self.proj_meta = self._load_json(project_info_path) if project_info_path else {}
        
        self.views_map = self._load_views_map(views_csv_path)

        floor_config = self._load_json(floor_map_path).get('floor_map', {}) if floor_map_path else {}
        self.defined_floors = list(floor_config.keys()) if floor_config else []

    def _resolve_labels(self, loader, explicit_names=None):
        """统一处理 Label 优先级：Loader > Explicit Args > Self"""
        if hasattr(loader, 'target_class_names') and loader.target_class_names:
            return loader.target_class_names
        if explicit_names:
            return explicit_names
        return self.labels

    def _load_json(self, path):
        if not path or not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)

    def _load_views_map(self, path):
        if not path or not os.path.exists(path): return {}
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            if 'view' in cols and 'direction' in cols:
                return dict(zip(df[cols['view']], df[cols['direction']]))
        except Exception as e: 
            print(f"[Warn] Failed to load views map: {e}")
        return {}

    def _enrich_data(self, df, view_name):
        """注入物理信息 (Floor, Real H) - 逻辑保持不变"""
        if df.empty: return df
        
        # 预计算 view 方向字符串，避免循环内重复查找
        ele_str = self.views_map.get(view_name.strip(), self.views_map.get(view_name, "Unknown"))
        
        # 使用 apply 加速或者是 list comprehension (此处保持原有逻辑结构，仅微调)
        floors, ids, orientations = [], [], []
        
        # 优化：提前获取 view 对应的 meta list，避免字典查找开销
        # 注意：这里假设 proj_meta 的 key 是 img_name
        
        for idx, row in df.iterrows():
            id = row['id']
            img_name = row['img_name']
            
            fl = "N/A"
            # 只有当 img_name 在 meta 中才遍历
            if img_name in self.proj_meta:
                # 这里通常是一个小列表，循环尚可
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
        # 1. 目录初始化
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis_fuse') 
        self.crop_dir = os.path.join(self.base_dir, 'report_crop_fuse')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        raw_data = self.loader.load()
        if not raw_data: return

        # 2. 处理图像
        # 传递 self.exif_db 确保 Processor 能获取 GPS
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir, exif_db=self.exif_db)
        
        raw_results = [None] * len(raw_data)
        print(f"Processing View: {view_name}...")

        if use_multithreading:
            print(f"[{time.strftime('%H:%M:%S')}] Starting multi-threaded processing ({max_workers} workers)...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {executor.submit(processor.process, item): i for i, item in enumerate(raw_data)}
                for future in tqdm(as_completed(future_to_index), total=len(raw_data), desc="Processing Images"):
                    idx = future_to_index[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as e:
                        print(f"❌ Error processing image index {idx}: {e}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Starting single-threaded processing...")
            for i, item in enumerate(tqdm(raw_data, desc="Processing Images")):
                raw_results[i] = processor.process(item)

        # 3. 后处理 (Enrich Data)
        all_dfs = [self._enrich_data(df, view_name) for df in raw_results if df is not None and not df.empty]

        if not all_dfs:
            print("No defects found.")
            return

        # 4. 组织与导出
        final_records = pd.concat(all_dfs, ignore_index=True).sort_values(by=['ID', 'floor']) if style_id == 3 else all_dfs
        
        # 复用 full_df 逻辑
        full_df = final_records if isinstance(final_records, pd.DataFrame) else pd.concat(all_dfs, ignore_index=True)
        unique_ids = full_df['ID'].nunique() if not full_df.empty else 0
        
        report_data = {
            'input': {'number': len(raw_data), 'shape': (0,0,0,0), 'type': f'{view_name}'},
            'output': {
                'model': model_name, 
                'defects': unique_ids, 
                'no defects': 0, 
                'defects sta': full_df.drop_duplicates(subset=['ID'])['Category'].value_counts().to_dict(),
                'elevation': self.views_map.get(view_name, '')
            },
            'records': [final_records] if style_id == 3 else final_records, # Style 3 expects list containing one DF
            'defined_categories': self.labels,
            'defined_floors': self.defined_floors
        }

        ExporterClass = EXPORTER_MAP.get(style_id)
        if ExporterClass:
            ExporterClass().export(report_data, output_path)
            print(f"Report Generated: {output_path}")


class BatchDedupEngine(DedupReportEngine):
    """
    派生类：用于批量处理 View 并生成汇总报告。
    """
    def __init__(self, exif_db, views_csv_path=None, floor_map_path=None, labels=None):
        """
        初始化批量引擎。
        只需传入全局通用的数据库和映射表，无需传入具体的 loader 或 project_path。
        """
        super().__init__(
            loader=None,              # 批量模式初始不需要 loader
            labels=labels,            # 可选，如果为空则会在 process 循环中从 loader 获取
            project_info_path=None,   # 批量模式下动态加载，初始为空
            group_info_path=None,     # 不需要
            views_csv_path=views_csv_path,
            floor_map_path=floor_map_path,
            exif_db=exif_db
        )

    def _process_view_generic(self, view_id, loader_cls, loader_kwargs, processor_cls, output_dirs, max_workers=1):
        """
        [核心合并方法] 统一处理普通 View 和 Aux View (支持多线程)
        """
        print(f"--- [Batch] Collecting data for {view_id} ---")

        # 1. 动态实例化 Loader
        loader_kwargs['exif_db'] = self.exif_db
        current_loader = loader_cls(**loader_kwargs)

        # 2. 更新 Labels
        new_labels = self._resolve_labels(current_loader)
        if new_labels:
            self.labels = new_labels

        # 3. 加载原始数据
        raw_data = current_loader.load()
        if not raw_data:
            print(f"    No data found for {view_id}")
            return pd.DataFrame()

        # 4. 确保目录存在
        for path in output_dirs.values():
            if path: os.makedirs(path, exist_ok=True)

        # 5. 初始化 Processor
        proc_kwargs = {
            'labels': self.labels,
            'colors': config.COLOR_PALETTE,
            'vis_dir': output_dirs['vis'],
            'crop_dir': output_dirs['crop'],
            'exif_db': self.exif_db
        }
        if 'vis_aux' in output_dirs: proc_kwargs['vis_aux_dir'] = output_dirs['vis_aux']
        if 'crop_aux' in output_dirs: proc_kwargs['crop_aux_dir'] = output_dirs['crop_aux']

        processor = processor_cls(**proc_kwargs)

        # 6. 处理数据 (核心差异部分)
        # 预分配列表以保持原始顺序 (这对于调试或后续按文件名排序很重要)
        raw_results = [None] * len(raw_data)

        if max_workers>1:
            # --- 多线程模式 ---
            # print(f"    Using {max_workers} threads...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_idx = {executor.submit(processor.process, item): i for i, item in enumerate(raw_data)}
                
                # 获取结果 (使用 tqdm 显示进度)
                for future in tqdm(as_completed(future_to_idx), total=len(raw_data), desc=f"    Analyzing {view_id} (Multi)", leave=False):
                    idx = future_to_idx[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as e:
                        print(f"    [Error] Failed to process image index {idx}: {e}")
        else:
            # --- 单线程模式 ---
            for i, item in enumerate(tqdm(raw_data, desc=f"    Analyzing {view_id} (Single)", leave=False)):
                try:
                    raw_results[i] = processor.process(item)
                except Exception as e:
                    print(f"    [Error] Failed to process image index {i}: {e}")

        # 7. 后处理：统一注入元数据 (Enrich Data)
        # 这部分很快且涉及共享资源读取，放在主线程串行执行最安全
        view_dfs = []
        for df in raw_results:
            if df is not None and not df.empty:
                # 注入楼层、物理高度等信息
                df = self._enrich_data(df, view_id)
                view_dfs.append(df)
        
        return pd.concat(view_dfs, ignore_index=True) if view_dfs else pd.DataFrame()

    def process_view_data(self, view_id, img_dir, label_dir, project_info_path, class_path, target_cls_ids=None, max_workers=1):
        """处理单图模式"""
        # 懒加载更新 Project Meta (仅在路径确实改变时重新加载，如果不常变可优化)
        if self.project_info_path != project_info_path:
            self.proj_meta = self._load_json(project_info_path)
            self.project_info_path = project_info_path
        
        # 准备目录
        # 注意：这里使用 label_dir 的相对路径，保持原逻辑
        base_output = os.path.dirname(label_dir) # 假设 label_dir 是 .../labels
        output_dirs = {
            'vis': os.path.join(base_output, 'batch_vis'),
            'crop': os.path.join(base_output, 'batch_crop')
        }

        loader_kwargs = {
            'img_dir': img_dir,
            'txt_dir': label_dir,
            'class_path': class_path,
            'target_cls_ids': target_cls_ids
        }

        return self._process_view_generic(
            view_id, 
            DedupLoader, 
            loader_kwargs, 
            DedupProcessor, 
            output_dirs,
            max_workers=max_workers,
        )

    def process_view_data_aux(self, view_id, img_dir, img_aux_dir, label_dir, project_info_path, class_path, target_cls_ids=None, max_workers=1):
        """处理双图 (Aux) 模式"""
        if self.project_info_path != project_info_path:
            self.proj_meta = self._load_json(project_info_path)
            self.project_info_path = project_info_path

        base_output = os.path.dirname(label_dir)
        output_dirs = {
            'vis': os.path.join(base_output, 'batch_vis'),
            'crop': os.path.join(base_output, 'batch_crop'),
            'vis_aux': os.path.join(base_output, 'batch_vis_aux'),
            'crop_aux': os.path.join(base_output, 'batch_crop_aux')
        }

        loader_kwargs = {
            'img_dir': img_dir,
            'img_aux_dir': img_aux_dir,
            'txt_dir': label_dir,
            'class_path': class_path,
            'target_cls_ids': target_cls_ids
        }

        return self._process_view_generic(
            view_id, 
            DedupLoader_AuxImage, 
            loader_kwargs, 
            DedupProcessor_AuxImage, 
            output_dirs,
            max_workers=max_workers,
        )

    def export_aggregated_report(self, all_df, output_path, model_name="BDD-MODEL", style_id=4):
        """导出汇总报告"""
        if all_df.empty:
            print("[ERROR] No aggregated data to export.")
            return

        print(f">>> [Batch] Generating Aggregated Report: {output_path}")
        
        # 1. 智能排序 (利用正则提取数字)
        if 'view' in all_df.columns:
            def extract_num(x):
                m = re.search(r'\d+', str(x))
                return int(m.group()) if m else 999
            
            # 使用临时列排序，避免修改原数据结构
            all_df['__view_sort'] = all_df['view'].apply(extract_num)
            all_df = all_df.sort_values(by=['__view_sort', 'ID'])
            all_df.drop(columns=['__view_sort'], inplace=True)

        # 2. 构建 Report Data
        unique_ids = all_df['ID'].nunique()
        # 兼容 view 可能是数字或字符串
        view_list = sorted(all_df['view'].astype(str).unique())
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

        # 3. 导出
        ExporterClass = EXPORTER_MAP.get(style_id)
        if ExporterClass: 
            ExporterClass().export(report_data, output_path)
            print(f"Done! Saved to {output_path}")

