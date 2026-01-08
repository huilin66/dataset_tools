import os
import config
from loaders.dedup_loader import DedupLoader
from core.dedup_engine import DedupReportEngine

def run_dedup_reporting(view_id="V30"):
    # === 1. 路径配置 (根据你的 yolo_dedup 输出结构) ===
    # 假设你的根目录结构如下：
    # split_data/
    #   ├── visible_views/V30/ (图片)
    #   └── thermal_views_infer_dedup/V30/ (Dedup 结果)
    
    ROOT_DATA = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data"
    views_path = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs\views_direction.csv'
    # 输入：图片目录
    img_dir = os.path.join(ROOT_DATA, "thermal_views", view_id) # 或者是 visible_views
    
    # 输入：Dedup 结果根目录
    dedup_root = os.path.join(ROOT_DATA, "thermal_views_infer_dedup", view_id)
    
    # 具体文件路径
    # 使用 labels_dedup_fuse 作为标签源，因为它包含合并后的框和 ID
    label_dir = os.path.join(dedup_root, "labels_dedup_fuse") 
    project_info = os.path.join(dedup_root, "project_info.json")
    group_info = os.path.join(dedup_root, "labels_group_info.json")
    
    # 输出：PDF 路径
    output_pdf = os.path.join(dedup_root, f"Report_{view_id}_Dedup.pdf")
    
    # 类别文件
    class_path = config.CLASS_PATH
    
    print(f">>> Starting Dedup Report for {view_id}")
    print(f"    Images: {img_dir}")
    print(f"    Labels: {label_dir}")
    
    # 初始化 DedupLoader
    # 它会读取 labels_dedup_fuse 中的 7 列数据
    loader = DedupLoader(img_dir=img_dir, txt_dir=label_dir, class_path=class_path, target_cls_ids=[0, 2])
    
    # === 3. 运行引擎 ===
    # 初始化 DedupEngine
    engine = DedupReportEngine(
        loader=loader, 
        labels=[],
        project_info_path=project_info, 
        group_info_path=group_info,
        views_csv_path=views_path,
        floor_map_path=r"E:\repository\dataset_tools\floor_map.json",
    )
    
    # 生成报告
    engine.run(output_path=output_pdf, view_name=view_id, style_id=4)

if __name__ == '__main__':
    # 示例：为 V30 生成报告
    run_dedup_reporting("V30")