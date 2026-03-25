# core/engine.py
import os
import time
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed # 用于多线程加速

# 内部模块调用
import config # 获取全局配置，如颜色、默认参数
from utils.metadata import MetadataManager, safe_float # 元数据管理与安全转换工具
from core.processor import ImageProcessor # 核心图像处理逻辑类
from exporters import EXPORTER_MAP # 报告样式映射表
from utils.analysis import img_sta # 图像尺寸统计工具

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