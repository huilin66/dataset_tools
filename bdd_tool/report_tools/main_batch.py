import os
import re
import pandas as pd
import config
from core.batch_engine import BatchDedupEngine  # 导入刚才创建的派生类

def natural_sort_key(s):
    """自然排序辅助函数: V2 < V10"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def run_batch_reporting_all_views(root_data_path, output_filename="Total_Project_Report.pdf"):
    # === 1. 路径定义 ===
    # Dedup 结果根目录 (包含 V01, V02...)
    dedup_base_dir = os.path.join(root_data_path, "thermal_views_infer_dedup")
    # 原始图片根目录 (包含 V01, V02...)
    img_base_dir = os.path.join(root_data_path, "thermal_views") 
    
    # 公共资源路径
    views_csv_path = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs\views_direction.csv'
    floor_map_path = r"E:\repository\dataset_tools\floor_map.json"
    
    # 最终输出路径
    final_output_pdf = os.path.join(dedup_base_dir, output_filename)
    
    # === 2. 扫描所有 View 文件夹 ===
    if not os.path.exists(dedup_base_dir):
        print(f"Error: Directory not found: {dedup_base_dir}")
        return

    # 查找所有 V 开头的文件夹
    all_views = [d for d in os.listdir(dedup_base_dir) 
                 if os.path.isdir(os.path.join(dedup_base_dir, d)) and d.upper().startswith('V')]
    
    # 排序 (V1, V2, ... V10)
    all_views.sort(key=natural_sort_key)
    
    print(f">>> Found {len(all_views)} views to process: {all_views}")

    # === 3. 初始化 Batch Engine ===
    # 注意：这里我们不传入具体的 loader，只传入公共配置
    # loader=None, labels=[] (labels会在engine内部处理或从config读取)
    engine = BatchDedupEngine(
        loader=None, 
        labels=[], 
        project_info_path="", # 暂时为空，循环中会动态加载
        group_info_path="",   # 暂时为空
        views_csv_path=views_csv_path,
        floor_map_path=floor_map_path
    )
    
    # 设置一个公共的资源输出目录，用于存放生成的 crop 图片 (可选)
    # 如果想让每个 View 的图片留在各自文件夹，这里可以不设置，依靠 process_view_data 内部逻辑
    engine.vis_dir = os.path.join(dedup_base_dir, "batch_report_assets", "vis")
    engine.crop_dir = os.path.join(dedup_base_dir, "batch_report_assets", "crop")
    os.makedirs(engine.vis_dir, exist_ok=True)
    os.makedirs(engine.crop_dir, exist_ok=True)

    # === 4. 循环收集数据 ===
    all_dfs = []
    
    for view_id in all_views:
        # 构造当前 View 的具体路径
        curr_img_dir = os.path.join(img_base_dir, view_id)
        curr_dedup_dir = os.path.join(dedup_base_dir, view_id)
        
        curr_label_dir = os.path.join(curr_dedup_dir, "labels_dedup_fuse")
        curr_proj_info = os.path.join(curr_dedup_dir, "project_info.json")
        
        # 检查必要文件是否存在
        if not os.path.exists(curr_label_dir):
            print(f"Skipping {view_id}: labels_dedup_fuse not found.")
            continue
            
        # 调用 engine 的收集方法
        df = engine.process_view_data(
            view_id=view_id,
            img_dir=curr_img_dir,
            label_dir=curr_label_dir,
            project_info_path=curr_proj_info
        )
        
        if not df.empty:
            all_dfs.append(df)
            
    # === 5. 合并与导出 ===
    if all_dfs:
        print(">>> Merging data from all views...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f'get {len(final_df)} rows of data after merging {len(all_dfs)}')

        # 调用 engine 的导出方法
        # 使用 Style 4 (PDFExporterWithContext)
        engine.export_aggregated_report(
            all_df=final_df, 
            output_path=final_output_pdf,
            style_id=4 
        )
    else:
        print("No data collected from any view.")

if __name__ == '__main__':
    # 配置你的根数据路径
    ROOT_DATA = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data"
    
    run_batch_reporting_all_views(ROOT_DATA)