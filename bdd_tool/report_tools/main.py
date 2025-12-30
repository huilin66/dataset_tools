# main.py
import os
import config
from pathlib import Path
from loaders.yolo_loader import YoloLoader
from core.engine import ReportEngine

def load_class_list(class_path):
    """加载类别定义文件"""
    if not os.path.exists(class_path):
        return []
    with open(class_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def dji_metadata_provider(img_path):
    """
    可选的外部元数据提供者。
    如果需要特定的楼层计算逻辑或从外部数据库读取位置，可在此扩展。
    """
    # 此处逻辑已在 Processor 中通过 MetadataManager 自动处理
    # 仅当你有额外的业务逻辑（如：根据经纬度反查楼宇名称）时才需在此编写并传给 Engine
    return None

def run_inspection_task(img_dir, label_dir, class_path, output_pdf, style_id=3):
    """
    执行单次巡检报告生成任务
    """
    # 1. 加载类别列表
    classes = load_class_list(class_path)
    
    # 2. 初始化数据加载器
    # 支持 yolo 格式加载
    loader = YoloLoader(img_dir=img_dir, txt_dir=label_dir, class_list=classes)

    # 3. 初始化报告引擎
    # 引擎现在内部持有 Processor 和 MetadataManager
    engine = ReportEngine(loader=loader, labels=classes)
    
    # 4. 运行引擎
    # 包含：元数据声明、图像解析、PDF导出、状态重置
    engine.run(
        output_path=output_pdf, 
        model_name="DJI-M4T-INSPECTION", 
        style_id=style_id
    )

if __name__ == '__main__':
    # ==========================================
    # 任务 1: 热成像数据组 (示例)
    # ==========================================
    print(">>> Starting Task: Thermal Inspection")
    run_inspection_task(
        img_dir=config.IMG_DIR, 
        label_dir=config.PRED_DIR, 
        class_path=config.CLASS_PATH, 
        output_pdf=config.OUTPUT_PDF_PATH,
        style_id=3  # 使用紧凑横向样式
    )

    # ==========================================
    # 任务 2: 可见光数据组 (如果需要连续运行)
    # ==========================================
    # print("\n>>> Starting Task: Visible Light Inspection")
    # run_inspection_task(
    #     img_dir=r'path/to/visible/images',
    #     label_dir=r'path/to/visible/labels',
    #     class_path=config.CLASS_PATH,
    #     output_pdf=r'path/to/output/visible_report.pdf',
    #     style_id=3
    # )