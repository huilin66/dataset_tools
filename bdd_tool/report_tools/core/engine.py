# core/engine.py
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from utils.metadata import MetadataManager, safe_float
from core.processor import ImageProcessor
from exporters import EXPORTER_MAP
from utils.analysis import img_sta
import config

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

    def run(self, output_path, model_name="BDD-MODEL", style_id=3):
        self.base_dir = os.path.dirname(os.path.abspath(output_path))
        self.vis_dir = os.path.join(self.base_dir, 'report_vis')
        self.crop_dir = os.path.join(self.base_dir, 'report_crop')
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.crop_dir, exist_ok=True)

        raw_data = self.loader.load()
        if not raw_data:
            print("No data found.")
            return

        self._declare_metadata(raw_data[0]['image_path'])

        processor = ImageProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir, self.metadata_getter)
        all_dfs = []
        img_paths = []
        
        for item in tqdm(raw_data, desc="Processing Images"):
            img_paths.append(item['image_path'])
            df = processor.process(item)
            all_dfs.append(df)

        # 数据聚合逻辑
        has_defect = sum(not df.empty for df in all_dfs)
        cat_counts = {}
        for df in all_dfs:
            if not df.empty:
                counts = df['Category'].value_counts()
                for cat, count in counts.items():
                    cat_counts[cat] = cat_counts.get(cat, 0) + count

        report_info = {
            'input': {'number': len(raw_data), 'shape': img_sta(img_paths), 'type': 'Images'},
            'output': {'model': model_name, 'defects': has_defect, 'no defects': len(raw_data)-has_defect, 'defects sta': cat_counts},
            'records': all_dfs,
            'drone_info': self.global_drone_info
        }

        ExporterClass = EXPORTER_MAP.get(style_id, EXPORTER_MAP[0])
        ExporterClass().export(report_info, output_path)
        
        self._init_metadata_store()
        print("--- Run Completed ---")