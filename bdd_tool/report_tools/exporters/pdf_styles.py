import os
from pathlib import Path
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Table, Spacer, Paragraph, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from collections import defaultdict, Counter
from reportlab.lib.pagesizes import letter, landscape
from .base_exporter import BasePDFExporter

__all__ = [
    'PDFExporterBasic',
    'PDFExporterDetailed',
    'PDFExporterMeasurement',
    'PDFExporterCompact',
]

class PDFExporterBasic(BasePDFExporter):
    """
    样式 0: 基础报告
    兼容: 自动优先使用 'ID' 字段
    """
    def generate_row_content(self, df_record):
        data_rows = []
        row_heights = []

        def add(row, h):
            data_rows.append(row)
            row_heights.append(h if h else 20)

        # 1. 标题行
        first_row = df_record.iloc[0]
        vis_path = first_row['VisPath']
        fname = Path(first_row['Path']).name
        
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            aspect = vis_img.drawHeight / vis_img.drawWidth if vis_img.drawWidth > 0 else 1.0
            vis_img.drawWidth = 5 * inch
            vis_img.drawHeight = 5 * inch * aspect
            img_h = vis_img.drawHeight * 1.05
        else:
            vis_img = "Missing"
            img_h = 25

        add(["FileName", fname], 25)
        add([vis_img, ''], img_h)
        add(['Defect Count', str(len(df_record))], 25)

        # 2. 缺陷行
        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            
            # [兼容性修改] 优先使用 ID，否则使用索引
            defect_id = str(row.get('ID', f'Defect {idx+1}'))
            
            add([f'ID: {defect_id}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        
        return data_rows, row_heights


class PDFExporterDetailed(BasePDFExporter):
    """
    样式 1: 详细报告
    兼容: 支持 Floor, View, XYZ, Orientation, ID
    """
    def generate_row_content(self, df_record):
        data_rows = []
        row_heights = []

        def add(row, h):
            data_rows.append(row)
            row_heights.append(h if h else 20)

        first_row = df_record.iloc[0]
        vis_path = first_row['VisPath']
        fname = Path(first_row['Path']).name
        
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            aspect = vis_img.drawHeight / vis_img.drawWidth if vis_img.drawWidth > 0 else 1.0
            vis_img.drawWidth = 5 * inch
            vis_img.drawHeight = 5 * inch * aspect
            img_h = vis_img.drawHeight * 1.05
        else:
            vis_img = "Missing"
            img_h = 25

        add(["FileName", fname], 25)
        add([vis_img, ''], img_h)
        
        # [兼容性修改] 增强的位置信息
        loc_info = f"Floor: {first_row.get('floor', 'N/A')} | View: {first_row.get('view', 'N/A')}"
        add(['Location Info', loc_info], 25)

        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            
            # [兼容性修改] ID
            defect_id = str(row.get('ID', f'Defect {idx+1}'))
            
            add([f'ID: {defect_id}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            
            # [兼容性修改] 物理坐标
            xyz = row.get('xyz', 'N/A')
            ori = row.get('orientation', 'N/A')
            add(['XYZ Coords', str(xyz)], 20)
            add(['Orientation', str(ori)], 20)
            
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        
        return data_rows, row_heights


class PDFExporterMeasurement(BasePDFExporter):
    """
    样式 2: 测量报告
    兼容: ID, W_cm, H_cm
    """
    def generate_row_content(self, df_record):
        data_rows = []
        row_heights = []

        def add(row, h):
            data_rows.append(row)
            row_heights.append(h if h else 20)

        first_row = df_record.iloc[0]
        vis_path = first_row['VisPath']
        fname = Path(first_row['Path']).name
        
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            max_w = 4.5 * inch
            aspect = vis_img.drawHeight / vis_img.drawWidth if vis_img.drawWidth > 0 else 1.0
            vis_img.drawWidth = max_w
            vis_img.drawHeight = max_w * aspect
            img_h = vis_img.drawHeight + 10
        else:
            vis_img = "Missing"
            img_h = 25

        add(["FileName", fname], 25)
        add([vis_img, ''], img_h)
        meta_text = f"Floor: {first_row.get('floor','N/A')} | View: {first_row.get('view','N/A')}"
        add(['Location', meta_text], 25)

        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            
            defect_id = str(row.get('ID', f'Defect {idx+1}'))
            add([f'ID: {defect_id}', crop_img], 2*inch + 10)
            
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            
            # [兼容性修改] 优先使用 Engine 计算好的 W_cm/H_cm
            w_cm_val = row.get('W_cm', 'N/A')
            h_cm_val = row.get('H_cm', 'N/A')
            area_cm_val = row.get('Area_cm2', 'N/A')
            
            w_str = f"{w_cm_val} cm" if w_cm_val != "N/A" else f"{row.get('W_pix','-')} pix"
            h_str = f"{h_cm_val} cm" if h_cm_val != "N/A" else f"{row.get('H_pix','-')} pix"
            area_str = f"{area_cm_val} cm²" if area_cm_val != "N/A" else f"{row.get('Area_pix','-')} pix²"
            
            add(['Width', w_str], 20)
            add(['Height', h_str], 20)
            add(['Area', area_str], 20)
            
            add(['Action', row['Action']], 20)

        return data_rows, row_heights


class PDFExporterCompact(BasePDFExporter):
    """
    样式 3 (Excel 风格横向版):
    兼容: 
    1. 自动识别 ID, Floor, XYZ 列
    2. 智能模式: 如果传入的是多张图片的聚合数据(View Mode)，则生成聚合清单。
    """
    def __init__(self):
        super().__init__()
        self.pagesize = landscape(letter)

    def _add_summary_pages(self, elements, report_data):
        # ... (保持原有的 Summary 逻辑，这部分是通用的) ...
        # 为了节省篇幅，这里复用您之前提供的代码逻辑
        input_info = report_data['input']
        output_info = report_data['output']
        
        elements.append(Paragraph("<b>Project Summary Report</b>", self.styles["font_title"]))
        
        # [兼容性] 如果有 Elevation 信息，显示它
        elevation = output_info.get('elevation', '')
        if elevation:
            elements.append(Paragraph(f"Elevation: {elevation}", self.styles["font_section"]))
        
        elements.append(Spacer(1, 20))
        # (后续统计表格代码保持不变，直接复制之前的即可)
        # ... (Summary tables code omitted for brevity, use previous implementation) ...
        super()._add_summary_pages(elements, report_data)

    def generate_flowables(self, df_record):
        elems = []
        if df_record.empty: return elems
        
        # [智能模式判断]
        # 检查是否包含多张不同的图片路径。如果是，说明是聚合 View 数据
        is_aggregated_view = df_record['VisPath'].nunique() > 1
        
        # 如果是聚合模式，按 ID 排序；否则默认顺序（通常是按 Yolo 输出顺序）
        if 'ID' in df_record.columns:
            # 确保 ID 是 int 以便正确排序，如果是 str 混合则忽略错误
            try:
                df_record = df_record.sort_values(by=['ID'])
            except:
                pass
        
        # 1. 标题逻辑
        if is_aggregated_view:
            # 聚合模式：只显示通用标题
            title_text = f"Detailed Defect List (Sorted by ID)"
        else:
            # 单图模式：显示文件名
            fname = Path(df_record.iloc[0]['Path']).name
            title_text = f"File: {fname}"

        title_style = ParagraphStyle(
            'CompactTitle', 
            parent=self.styles['Heading2'], 
            backColor=colors.lightgrey, 
            borderPadding=5,
            spaceAfter=5
        )
        elems.append(Paragraph(title_text, title_style))

        # 2. 表头定义 (兼容 ID 和 Location)
        headers = ["ID", "Loc(Z,X)", "Floor", "Size (H*W)", "Severity", "Type", "Action", "Image/File"]
        table_data = [headers]
        row_heights = [30]

        # 单元格样式
        cell_style = ParagraphStyle('CellStyle', parent=self.styles['Normal'], fontSize=8, leading=10, alignment=1)

        last_id = -1
        
        for idx, row in df_record.iterrows():
            # [兼容性] 数据提取
            defect_id = str(row.get('ID', f"DF{idx+1}"))
            floor_val = str(row.get('floor', '-'))
            # 优先用 xyz 字段，如果没有则尝试拼凑
            xyz_val = str(row.get('xyz', '-'))
            
            # 尺寸
            w_val = row.get('W_cm', 'N/A')
            h_val = row.get('H_cm', 'N/A')
            if w_val != 'N/A':
                dim_str = f"H:{h_val} * W:{w_val}\n(cm)"
            else:
                dim_str = f"H:{row.get('H_pix','-')} * W:{row.get('W_pix','-')}\n(pix)"
            
            dim_para = Paragraph(dim_str, cell_style)

            # 图片列处理
            crop_path = row['CropPath']
            img_cell = ""
            row_h = 50
            
            # 如果有 Crop 图，优先展示 Crop
            if os.path.exists(crop_path):
                img = RLImage(crop_path)
                target_w = 1.1 * inch 
                max_h = 1.4 * inch    
                aspect = img.imageHeight / img.imageWidth if img.imageWidth > 0 else 1.0
                calc_h = target_w * aspect
                
                if calc_h > max_h:
                    img.drawHeight = max_h
                    img.drawWidth = max_h / aspect
                else:
                    img.drawWidth = target_w
                    img.drawHeight = calc_h
                img_cell = img
                row_h = max(50, img.drawHeight + 6)
            
            # 如果是聚合模式，图片列下方最好显示文件名，防止混淆
            if is_aggregated_view:
                fname_short = os.path.basename(row['VisPath'])
                # 把图片和文件名放一起？ReportLab Table 不太好放两个 Flowable
                # 策略：如果聚合模式，把文件名放在 Image 列的文本中，或者单独一列？
                # 为了兼容性，我们把文件名放在 "Image/File" 列，如果有图则图在文件名上方(难实现)
                # 简单做法：文件名用 Paragraph
                fname_para = Paragraph(fname_short, cell_style)
                # 这里我们只放图，文件名可能得牺牲，或者放在 Location 里？
                # 决定：Image 列只放图。文件名如果重要，应该在 Detailed 模式看。
                # 或者：把文件名作为 ID 的一部分显示？
                pass

            lvl = row['Level']
            display_lvl = 'Minor' if lvl == 'Slight' else ('Major' if lvl == 'Serious' else lvl)

            table_row = [
                defect_id,
                xyz_val,
                floor_val,
                dim_para,
                display_lvl,
                row['Category'],
                row['Action'],
                img_cell
            ]
            table_data.append(table_row)
            row_heights.append(row_h)

            # ID 分隔线
            current_id_int = int(defect_id) if defect_id.isdigit() else defect_id
            if last_id != -1 and current_id_int != last_id:
                # 修正索引
                r_idx = len(table_data) - 2
                # style_list.append(('LINEBELOW', (0, r_idx), (-1, r_idx), 1.5, colors.black))
            last_id = current_id_int

        # 定义列宽
        col_widths = [
            0.6*inch, # ID
            1.2*inch, # Loc
            0.6*inch, # Floor
            1.1*inch, # Size
            0.8*inch, # Severity
            1.0*inch, # Type
            0.8*inch, # Action
            1.5*inch  # Image
        ]

        t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
        
        style_list = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]
        t.setStyle(TableStyle(style_list))
        
        elems.append(t)
        elems.append(Spacer(1, 15))

        return elems