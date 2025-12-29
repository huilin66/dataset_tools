# -*- coding: utf-8 -*-
"""
YOLO Report Generator (Folder Mode)
读取图片文件夹和YOLO格式的Txt预测文件夹，生成PDF分析报告。
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch

# -------------------------------------------------------------------------
# 1. Visual Utils (绘制与裁剪)
# -------------------------------------------------------------------------

def draw_box(img_pil, bboxes, labels, colors):
    """
    在 PIL 图片上绘制边界框
    bboxes: [[cls, score, x1, y1, x2, y2], ...] (Pixel coordinates)
    """
    draw = ImageDraw.Draw(img_pil)
    try:
        # 尝试使用系统字体，如果失败使用默认
        # Linux/Mac 用户可能需要调整路径，例如 "/usr/share/fonts/truetype/..."
        font = ImageFont.truetype("arial.ttf", size=max(15, int(img_pil.width/50)))
    except IOError:
        font = ImageFont.load_default()

    for box in bboxes:
        cls_id = int(box[0])
        score = float(box[1])
        x1, y1, x2, y2 = box[2:]
        
        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)

        label_text = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        color = colors[cls_id % len(colors)]
        
        if isinstance(color, str):
            fill_color = color
        else:
            fill_color = tuple(color[::-1]) # BGR -> RGB assuming tuple input

        # 画框
        line_width = max(2, int(img_pil.width/300))
        draw.rectangle([x1, y1, x2, y2], outline=fill_color, width=line_width)
        
        # 画标签
        text_content = f"{label_text} {score:.2f}"
        
        # 计算文字背景框大小
        left, top, right, bottom = draw.textbbox((x1, y1), text_content, font=font)
        draw.rectangle((left, top, right, bottom), fill=fill_color)
        draw.text((x1, y1), text_content, fill="white", font=font)
    
    return img_pil

def crop_box(img_pil, bboxes):
    """
    裁剪出检测框对应的图片
    """
    crops = []
    for box in bboxes:
        x1, y1, x2, y2 = box[2:]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_pil.width, int(x2)), min(img_pil.height, int(y2))
        
        if x2 > x1 and y2 > y1:
            crop = img_pil.crop((x1, y1, x2, y2))
            crops.append(crop)
        else:
            crops.append(Image.new('RGB', (50, 50), color='black'))
    return crops

# -------------------------------------------------------------------------
# 2. Data Processing & Logic
# -------------------------------------------------------------------------

levels_threshold = [50, 500]

def level_judge(box):
    # 简单的基于尺寸的等级判断逻辑
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    if w > levels_threshold[1] or h > levels_threshold[1]:
        return 'Serious'
    elif w > levels_threshold[0] or h > levels_threshold[0]:
        return 'Moderate'
    return 'Slight'

def action_judge(level):
    return 'Repair' if level in ['Serious', 'Moderate'] else 'Monitor'

def img_sta(img_paths):
    """统计图片尺寸"""
    if not img_paths: return [0, 0, 0, 0]
    shape_dict = {}
    for img_path in img_paths:
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                shape_dict[os.path.basename(img_path)] = img.size
    if not shape_dict: return [0, 0, 0, 0]
    shapes = np.array(list(shape_dict.values()))
    maxs, mins = np.max(shapes, axis=0), np.min(shapes, axis=0)
    return [mins[0], maxs[0], mins[1], maxs[1]]

def process_single_pair(image_path, detections, label_list, save_dir_vis, save_dir_crop, color_list):
    """
    处理单张图片及其对应的检测结果
    detections: [[cls, score, x1, y1, x2, y2], ...] (Pixel Coords)
    """
    stem_name = Path(image_path).stem
    os.makedirs(save_dir_vis, exist_ok=True)
    crop_subdir = os.path.join(save_dir_crop, stem_name)
    os.makedirs(crop_subdir, exist_ok=True)

    img = Image.open(image_path).convert('RGB')
    
    # 1. 保存可视化大图
    img_vis = img.copy()
    if len(detections) > 0:
        img_vis = draw_box(img_vis, detections, label_list, color_list)
    vis_path = os.path.join(save_dir_vis, stem_name + '.png')
    img_vis.save(vis_path)

    # 2. 裁剪小图并记录数据
    records = []
    crops = crop_box(img, detections)
    
    for i, bbox in enumerate(detections):
        cls_id = int(bbox[0])
        score = float(bbox[1])
        box = bbox[2:] # x1, y1, x2, y2
        
        cat_name = label_list[cls_id] if cls_id < len(label_list) else f"Class_{cls_id}"
        level = level_judge(box)
        
        # 保存裁剪图
        crop_filename = f"{stem_name}_{i}_{cls_id}.png"
        crop_path = os.path.join(crop_subdir, crop_filename)
        crops[i].save(crop_path)

        record = {
            'Path': image_path,
            'VisPath': vis_path,
            'CropPath': crop_path,
            'Category': cat_name.title(),
            'Level': level,
            'Score': score,
            'Bbox': str(list(box)),
            'Action': action_judge(level),
        }
        records.append(record)
    
    return pd.DataFrame(records)

# -------------------------------------------------------------------------
# 3. PDF Generation (ReportLab)
# -------------------------------------------------------------------------

# --- 样式定义开始 ---
# 字体回退机制
try:
    # 尝试加载 Windows 常用字体
    pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\\Windows\\Fonts\\times.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\\Windows\\Fonts\\timesbd.ttf"))
    FONT_REGULAR, FONT_BOLD = "TimesNewRoman", "TimesNewRoman-Bold"
except:
    # 回退到内置字体
    FONT_REGULAR, FONT_BOLD = "Helvetica", "Helvetica-Bold"

styles = getSampleStyleSheet()

# 确保添加了 'font_title', 'font_section', 'font_text' 这三个样式
styles.add(ParagraphStyle(name="font_title", fontName=FONT_BOLD, fontSize=22, alignment=1, leading=33))
styles.add(ParagraphStyle(name="font_section", fontName=FONT_BOLD, fontSize=20, leading=30))
styles.add(ParagraphStyle(name="font_text", fontName=FONT_REGULAR, fontSize=16, leading=24))

# 表格样式
threeline_table = TableStyle([
    ("LINEABOVE", (0, 0), (-1, 0), 2, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
    ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
    ("FONTSIZE", (0, 0), (-1, -1), 14),
    ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
])

blank_table = TableStyle([
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
    ("FONTSIZE", (0, 0), (-1, -1), 16),
])

new_threeline_table = TableStyle(threeline_table.getCommands())
# --- 样式定义结束 ---

def create_report_pdf(report_data, save_path):
    """
    生成 PDF 报告的主函数
    包含：样式定义、行高自动修正、进度条显示
    """
    print(f"[{time.strftime('%H:%M:%S')}] Report generation started...")
    
    # --- 1. 样式定义 (防止 KeyError) ---
    try:
        pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\\Windows\\Fonts\\times.ttf"))
        pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\\Windows\\Fonts\\timesbd.ttf"))
        FONT_REGULAR, FONT_BOLD = "TimesNewRoman", "TimesNewRoman-Bold"
    except:
        FONT_REGULAR, FONT_BOLD = "Helvetica", "Helvetica-Bold"

    styles = getSampleStyleSheet()
    # 显式添加自定义样式
    if 'font_title' not in styles:
        styles.add(ParagraphStyle(name="font_title", fontName=FONT_BOLD, fontSize=22, alignment=1, leading=33))
    if 'font_section' not in styles:
        styles.add(ParagraphStyle(name="font_section", fontName=FONT_BOLD, fontSize=20, leading=30))
    if 'font_text' not in styles:
        styles.add(ParagraphStyle(name="font_text", fontName=FONT_REGULAR, fontSize=16, leading=24))

    # 表格样式
    threeline_table = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 2, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
    ])

    blank_table = TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 16),
    ])
    
    new_threeline_table = TableStyle(threeline_table.getCommands())

    # --- 2. 准备文档结构 ---
    input_info = report_data['input']
    output_info = report_data['output']
    records_df_list = report_data['records']

    doc = SimpleDocTemplate(save_path, pagesize=letter)
    elements = []
    
    # --- Part 1 & 2: 摘要信息 ---
    print("Step 1/3: Preparing summary sections...")
    elements.append(Paragraph("<b>AI-Detection Result Report</b>", styles["font_title"]))
    elements.append(Spacer(1, 30))

    elements.append(Paragraph("Input Information:", styles["font_section"]))
    shape_str = f"{input_info['shape'][0]}~{input_info['shape'][1]}, {input_info['shape'][2]}~{input_info['shape'][3]}"
    data_input = [
        ["Type of Data:", input_info['type'].title()],
        ["Number of Images:", input_info['number']],
        ["Shape Range (W, H):", shape_str]
    ]
    t_input = Table(data_input, hAlign='LEFT', rowHeights=25)
    t_input.setStyle(blank_table)
    elements.append(t_input)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Detection Summary:", styles["font_section"]))
    data_output_summary = [
        ["Model Used:", output_info['model']],
        ["Images with Defects:", output_info['defects']],
        ["Images without Defects:", output_info['no defects']],
    ]
    t_summary = Table(data_output_summary, hAlign='LEFT', rowHeights=25)
    t_summary.setStyle(blank_table)
    elements.append(t_summary)
    elements.append(Spacer(1, 10))

    data_output_defects = [["Category", "Count"]]
    for k, v in output_info['defects sta'].items():
        data_output_defects.append([k.title(), v])
    t_stats = Table(data_output_defects, hAlign='CENTER', rowHeights=25)
    t_stats.setStyle(threeline_table)
    elements.append(t_stats)
    elements.append(PageBreak())

    # --- Part 3: 详细记录 (带进度条) ---
    print("Step 2/3: Assembling detailed records (this builds the table structure)...")
    elements.append(Paragraph("Detailed Information:", styles["font_section"]))
    elements.append(Spacer(1, 10))

    data_records = []
    rows_h = []

    # 辅助函数：安全添加行
    def add_row(row_data, height):
        data_records.append(row_data)
        if height is None: height = 20
        rows_h.append(height)

    # === 使用 tqdm 显示循环进度 ===
    # desc="Building Pages" 会显示在进度条左侧
    for df_record in tqdm(records_df_list, desc="Processing Images", unit="img"):
        if df_record.empty:
            continue
        
        first_row = df_record.iloc[0]
        vis_path = first_row['VisPath']
        file_name = Path(first_row['Path']).name
        
        # 主图处理
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            if vis_img.drawWidth > 0:
                aspect = vis_img.drawHeight / vis_img.drawWidth
            else:
                aspect = 1.0
            vis_img.drawWidth = 5 * inch
            vis_img.drawHeight = 5 * inch * aspect
            img_h = vis_img.drawHeight * 1.05
        else:
            vis_img = "Image Not Found"
            img_h = 25 

        # 添加表头行
        add_row(["FileName", file_name], 25)
        add_row([vis_img, ''], img_h)
        add_row(['Number of Defects', str(len(df_record))], 25)

        # 添加缺陷明细行
        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            if os.path.exists(crop_path):
                crop_img = RLImage(crop_path)
                crop_img.drawWidth = 2 * inch
                crop_img.drawHeight = 2 * inch
                crop_h = 2 * inch + 10
            else:
                crop_img = "Crop Not Found"
                crop_h = 25

            score_val = f"{row['Score']:.2f}" if 'Score' in row else "N/A"
            
            add_row([f'Defect {idx+1}', crop_img], crop_h)
            add_row(['Category', row['Category']], 20)
            add_row(['Level', row['Level']], 20)
            add_row(['Action', row['Action']], 20)
            add_row(['Score', score_val], 20)
            
    # 构建大表
    if data_records:
        # 再次检查长度一致性
        if len(data_records) != len(rows_h):
            min_len = min(len(data_records), len(rows_h))
            data_records = data_records[:min_len]
            rows_h = rows_h[:min_len]

        t_records = Table(data_records, hAlign='CENTER', colWidths=[2*inch, 4*inch], rowHeights=rows_h)
        
        final_style = TableStyle(new_threeline_table.getCommands())
        
        # 这里的循环非常快，一般不需要进度条，但为了保险起见也可以加
        for i, row_data in enumerate(data_records):
            if row_data[0] == 'FileName':
                final_style.add('SPAN', (0, i), (-1, i))
                final_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
            
        t_records.setStyle(final_style)
        elements.append(t_records)
    else:
        elements.append(Paragraph("No defects detected.", styles["font_text"]))

    # --- Part 4: 写入磁盘 ---
    print(f"Step 3/3: Rendering PDF layout and writing to disk...")
    print(f"Target path: {save_path}")
    print("Please wait, this might take a few seconds for large files...")
    
    try:
        doc.build(elements)
        print(f"[{time.strftime('%H:%M:%S')}] Success! Report saved.")
    except Exception as e:
        print(f"\n[ERROR] PDF Generation Failed: {e}")
        import traceback
        traceback.print_exc()

# -------------------------------------------------------------------------
# 4. YOLO IO Helper (核心修改部分)
# -------------------------------------------------------------------------

def yolo_norm_to_pixel(yolo_line, img_w, img_h):
    """
    将 YOLO txt的一行转换为像素坐标
    Input: "class x_center y_center w h [conf]" (Normalized 0-1)
    Output: [class, conf, x1, y1, x2, y2]
    """
    parts = yolo_line.strip().split()
    cls_id = int(parts[0])
    
    # 检查是否包含置信度
    if len(parts) >= 6:
        # 格式: class x y w h conf
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5])
    else:
        # 格式: class x y w h (无置信度，默认为 1.0)
        xc, yc, w, h = map(float, parts[1:5])
        conf = 1.0

    # 反归一化
    x_center = xc * img_w
    y_center = yc * img_h
    width = w * img_w
    height = h * img_h
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [cls_id, conf, x1, y1, x2, y2]

def load_data_from_folders(img_dir, txt_dir):
    """
    加载图片文件夹和TXT文件夹，进行配对
    """
    data_list = []
    
    # 支持多种图片格式
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        # 大小写兼容
        img_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    
    img_paths = sorted(list(set(img_paths))) # 去重排序

    print(f"Found {len(img_paths)} images in {img_dir}")

    for img_path in img_paths:
        stem = Path(img_path).stem
        txt_path = os.path.join(txt_dir, stem + '.txt')
        
        detections = []
        
        # 读取图片获取尺寸 (用于反归一化)
        with Image.open(img_path) as img:
            w, h = img.size
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        # 解析 YOLO 格式
                        det = yolo_norm_to_pixel(line, w, h)
                        detections.append(det)
        
        data_list.append({
            'image_path': img_path,
            'detections': np.array(detections) if detections else np.array([])
        })
        
    return data_list

# -------------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------------

class YoloReportGenerator:
    def __init__(self, labels, colors=None):
        self.labels = labels
        self.colors = colors if colors else [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)] * 10
    
    def run(self, input_data, output_pdf_path, model_name="YOLO-Model"):
        # 准备路径
        base_dir = os.path.dirname(os.path.abspath(output_pdf_path))
        vis_dir = os.path.join(base_dir, 'report_vis')
        crop_dir = os.path.join(base_dir, 'report_crop')
        
        all_results_dfs = []
        img_paths_list = []

        print("Processing images and generating visualizations...")
        for item in tqdm(input_data, desc="Processing"):
            img_path = item['image_path']
            dets = item['detections']
            img_paths_list.append(img_path)
            
            df = process_single_pair(img_path, dets, self.labels, vis_dir, crop_dir, self.colors)
            all_results_dfs.append(df)

        # 统计分析
        total = len(input_data)
        has_defect = sum(1 for df in all_results_dfs if not df.empty)
        
        cat_counts = {}
        for df in all_results_dfs:
            if not df.empty:
                counts = df['Category'].value_counts()
                for cat, count in counts.items():
                    cat_counts[cat] = cat_counts.get(cat, 0) + count

        report_info = {
            'input': {
                'number': total,
                'shape': img_sta(img_paths_list),
                'type': 'Images'
            },
            'output': {
                'model': model_name,
                'defects': has_defect,
                'no defects': total - has_defect,
                'defects sta': cat_counts
            },
            'records': all_results_dfs
        }

        print("Generating PDF...")
        create_report_pdf(report_info, output_pdf_path)

def load_class_list(class_path):
    with open(class_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="YOLO Result to PDF Report Generator")
    # parser.add_argument('--img_dir', type=str, required=True, help="Path to folder containing images")
    # parser.add_argument('--txt_dir', type=str, required=True, help="Path to folder containing YOLO format .txt results")
    # parser.add_argument('--out', type=str, default='report.pdf', help="Output PDF path")
    # parser.add_argument('--labels', type=str, default=None, help="Comma separated list of class names (e.g. 'crack,rust')")
    
    # args = parser.parse_args()

    # # 1. 设置标签
    # if args.labels:
    #     label_list = [l.strip() for l in args.labels.split(',')]
    # else:
    #     # 如果未提供，使用默认通用占位符
    #     print("Warning: No labels provided, using generic Class_0, Class_1...")
    #     label_list = [f"Class_{i}" for i in range(80)]

    root_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12'
    image_dir = os.path.join(root_dir, 'val', 'images')
    pred_dir = os.path.join(root_dir, 'result_analysis', 'val_infer', 'labels')
    class_path = os.path.join(root_dir, 'classes.txt')
    output_path = os.path.join(root_dir, 'result_analysis', 'val_infer', 'report.pdf')



    data = load_data_from_folders(image_dir, pred_dir)

    class_list = load_class_list(class_path)
    
    generator = YoloReportGenerator(labels=class_list)

    generator.run(data, output_pdf_path=output_path)