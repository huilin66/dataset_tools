import os
import pandas as pd
import json
from tqdm import tqdm
import re

# 导入原有类
from core.dedup_engine import DedupReportEngine, DedupProcessor
from loaders.dedup_loader import DedupLoader
from exporters import EXPORTER_MAP
import config

class BatchDedupEngine(DedupReportEngine):
    """
    派生类：用于批量处理 View 并生成汇总报告。
    """
    
    def __init__(self, *args, **kwargs):
        # 允许初始化时不传 loader，后续动态加载
        if 'loader' not in kwargs:
            kwargs['loader'] = None 
        super().__init__(*args, **kwargs)

    def process_view_data(self, view_id, img_dir, label_dir, project_info_path):
        """
        [核心扩展方法]
        处理单个 View，返回 DataFrame，但不生成 PDF。
        """
        print(f"--- [Batch] Collecting data for {view_id} ---")
        
        # 1. 动态实例化 Loader
        current_loader = DedupLoader(
            img_dir=img_dir, 
            txt_dir=label_dir, 
            class_path=config.CLASS_PATH, 
            target_cls_ids=[0, 2] 
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
        if not hasattr(self, 'vis_dir'):
            self.vis_dir = os.path.join(label_dir, '../batch_vis')
            self.crop_dir = os.path.join(label_dir, '../batch_crop')
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.crop_dir, exist_ok=True)

        # 5. 处理图像 (确保传入了修复后的 self.labels)
        processor = DedupProcessor(self.labels, config.COLOR_PALETTE, self.vis_dir, self.crop_dir)
        
        view_dfs = []
        for item in tqdm(raw_data, desc=f"    Analyzing {view_id}", leave=False):
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