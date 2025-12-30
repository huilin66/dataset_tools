# exporters/pdf_styles.py
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
    样式 0: 基础报告 (只包含图片、类别、等级、建议)
    """
    def generate_row_content(self, df_record):
        data_rows = []
        row_heights = []

        # 辅助函数
        def add(row, h):
            data_rows.append(row)
            row_heights.append(h if h else 20)

        # 1. 标题行
        first_row = df_record.iloc[0]
        vis_path = first_row['VisPath']
        fname = Path(first_row['Path']).name
        
        # 处理主图
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
            
            # --- 样式 0 的特定字段 ---
            add([f'Defect {idx+1}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        
        return data_rows, row_heights


class PDFExporterDetailed(BasePDFExporter):
    """
    样式 1: 详细报告 (增加 XYZ, Orientation, Floor, View)
    """
    def generate_row_content(self, df_record):
        data_rows = []
        row_heights = []

        def add(row, h):
            data_rows.append(row)
            row_heights.append(h if h else 20)

        # 1. 标题行 (同上，或者可以稍微变一下)
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
        # 可以在这里额外显示整张图的元数据
        add(['Location Info', f"Floor: {first_row.get('floor', 'N/A')} | View: {first_row.get('view', 'N/A')}"], 25)

        # 2. 缺陷行
        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            
            add([f'Defect {idx+1}', crop_img], 2*inch + 10)
            
            # --- 样式 1 的特定字段 (新增内容) ---
            add(['Category', row['Category']], 20)
            
            # 获取元数据 (即使是 N/A)
            xyz = row.get('xyz', 'N/A')
            ori = row.get('orientation', 'N/A')
            
            # 用颜色标记重要信息 (可选)
            add(['XYZ Coords', str(xyz)], 20)
            add(['Orientation', str(ori)], 20)
            
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        
        return data_rows, row_heights


class PDFExporterMeasurement(BasePDFExporter):
    """
    样式 3: 包含 像素/CM 测量信息 + 无人机摘要
    """

    def _add_summary_pages(self, elements, report_data):
        """
        重写摘要页：在原有基础上增加 Drone Info Table，并补全统计表
        """
        input_info = report_data['input']
        output_info = report_data['output']
        drone_info = report_data.get('drone_info', {})

        elements.append(Paragraph("<b>AI-Detection Result Report</b>", self.styles["font_title"]))
        elements.append(Spacer(1, 20))

        # --- 1. Drone Info Table (新增) ---
        if drone_info:
            elements.append(Paragraph("Drone Information:", self.styles["font_section"]))
            data_drone = [
                ["Drone Model:", drone_info.get('Model', 'N/A')],
                ["Camera Source:", drone_info.get('Camera', 'N/A')],
                ["Firmware Ver:", drone_info.get('Firmware', 'N/A')],
            ]
            t_drone = Table(data_drone, hAlign='LEFT', rowHeights=25, colWidths=[150, 300])
            t_drone.setStyle(self.style_blank)
            elements.append(t_drone)
            elements.append(Spacer(1, 10))

        # --- 2. Input Info ---
        elements.append(Paragraph("Input Information:", self.styles["font_section"]))
        shape_str = f"{input_info['shape'][0]}~{input_info['shape'][1]}, {input_info['shape'][2]}~{input_info['shape'][3]}"
        data_input = [
            ["Type of Data:", input_info['type'].title()],
            ["Number of Images:", str(input_info['number'])],
            ["Shape Range:", shape_str]
        ]
        t_input = Table(data_input, hAlign='LEFT', rowHeights=25, colWidths=[150, 300])
        t_input.setStyle(self.style_blank)
        elements.append(t_input)
        elements.append(Spacer(1, 10))

        # --- 3. Detection Summary ---
        elements.append(Paragraph("Detection Summary:", self.styles["font_section"]))
        data_output = [
            ["Model Used:", output_info['model']],
            ["Images With Defects:", str(output_info['defects'])],
            ["Images Without Defects:", str(output_info['no defects'])],
        ]
        t_output = Table(data_output, hAlign='LEFT', rowHeights=25, colWidths=[150, 300])
        t_output.setStyle(self.style_blank)
        elements.append(t_output)
        elements.append(Spacer(1, 10))
        
        # --- 4. Stats (【修复点】：补回统计表格) ---
        if output_info.get('defects sta'):
            elements.append(Paragraph("Defect Statistics:", self.styles["font_section"]))
            data_stats = [["Category", "Count"]]
            for cat, count in output_info['defects sta'].items():
                data_stats.append([cat, str(count)])
            
            t_stats = Table(data_stats, hAlign='LEFT', rowHeights=25, colWidths=[150, 100])
            t_stats.setStyle(self.style_threeline)
            elements.append(t_stats)
            elements.append(Spacer(1, 10))

        from reportlab.platypus import PageBreak
        elements.append(PageBreak())

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
            # 限制一下最大宽度，防止图片过大
            max_w = 4.5 * inch
            aspect = vis_img.drawHeight / vis_img.drawWidth if vis_img.drawWidth > 0 else 1.0
            vis_img.drawWidth = max_w
            vis_img.drawHeight = max_w * aspect
            img_h = vis_img.drawHeight + 10 # 留一点边距
        else:
            vis_img = "Missing"
            img_h = 25

        add(["FileName", fname], 25)
        add([vis_img, ''], img_h)
        meta_text = f"Floor: {first_row.get('floor','N/A')} | View: {first_row.get('view','N/A')}"
        add(['Location', meta_text], 25)

        # 2. 缺陷详情
        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            
            add([f'Defect {idx+1}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            
            # --- 【修改点】：修改尺寸显示格式 (yy cm / xx pix) ---
            # 如果 cm 是 N/A，则只显示 N/A / xx pix，或者根据需求调整
            w_cm_val = row['W_cm']
            h_cm_val = row['H_cm']
            area_cm_val = row['Area_cm2']
            
            # 格式：10.5cm / 100pix
            w_str = f"{w_cm_val} cm / {row['W_pix']} pix" if w_cm_val != "N/A" else f"N/A / {row['W_pix']} pix"
            h_str = f"{h_cm_val} cm / {row['H_pix']} pix" if h_cm_val != "N/A" else f"N/A / {row['H_pix']} pix"
            area_str = f"{area_cm_val} cm² / {row['Area_pix']} pix²" if area_cm_val != "N/A" else f"N/A / {row['Area_pix']} pix²"
            
            add(['Width', w_str], 20)
            add(['Height', h_str], 20)
            add(['Area', area_str], 20)
            
            add(['Action', row['Action']], 20)

        return data_rows, row_heights


class PDFExporterCompact(BasePDFExporter):
    """
    样式 3 (Excel 风格横向版):
    包含 Summary 交叉统计表 + 横向详细列表
    """
    def __init__(self):
        super().__init__()
        # 【关键】设置页面为横向 (Landscape)
        self.pagesize = landscape(letter)

    def _add_summary_pages(self, elements, report_data):
        # ... (summary 代码与上一次回答保持一致，此处省略以节省空间，直接用上一次的代码即可) ...
        # 如果你需要我再次提供 Summary 部分代码，请告诉我，否则请保留上一次修改的 Summary 逻辑
        pass 
        # (请保留上一次回答中的 _add_summary_pages 和 _get_summary_table_style 方法)


    def _add_summary_pages(self, elements, report_data):
        input_info = report_data['input']
        output_info = report_data['output']
        records_list = report_data['records']

        # 标题
        elements.append(Paragraph("<b>Project Summary Report</b>", self.styles["font_title"]))
        elements.append(Spacer(1, 20))

        # 0. 基础信息
        data_input = [
            ["Total Images:", str(input_info['number']), "Model:", output_info['model']],
            ["Detected Defects:", str(output_info['defects']), "Data Range:", f"{input_info['shape'][0]}~{input_info['shape'][1]}"],
        ]
        t_input = Table(data_input, hAlign='LEFT', colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2.5*inch])
        t_input.setStyle(self.style_blank)
        elements.append(t_input)
        elements.append(Spacer(1, 20))

        # --- 数据聚合 ---
        stats_cat_lev = defaultdict(lambda: defaultdict(int))
        stats_ele_lev = defaultdict(lambda: defaultdict(int))
        
        # 你的截图中有 Cosmetic, Critical 等，这里定义完整的表头顺序
        # 如果你的数据里只有 Minor/Moderate/Major，可以只保留这三个
        DISPLAY_LEVELS = ['Minor', 'Moderate', 'Major'] 
        
        # 映射逻辑 (根据需要调整)
        LEVEL_MAP = {
            'Slight': 'Minor', 
            'Moderate': 'Moderate', 
            'Serious': 'Major'
        }

        all_categories = set()
        all_elevations = set()

        for df in records_list:
            if df.empty: continue
            view = df.iloc[0].get('view', 'Unknown').strip()
            all_elevations.add(view)

            for _, row in df.iterrows():
                cat = row['Category']
                raw_level = row['Level']
                level = LEVEL_MAP.get(raw_level, raw_level)
                
                all_categories.add(cat)
                stats_cat_lev[level][cat] += 1
                stats_ele_lev[view][level] += 1

        sorted_cats = sorted(list(all_categories))
        sorted_eles = sorted(list(all_elevations))

        # ==========================================
        # 1. 统计表 A: Defect Type & Severity (已翻转)
        # 行：Defect Type (Category)
        # 列：Severity (Minor, Moderate, Major)
        # ==========================================
        elements.append(Paragraph("Summary by Defect Type & Severity:", self.styles["font_section"]))
        
        # 表头: [Defect Type] + [Minor, Moderate, Major] + [Total]
        header_a = ["Defect Type"] + DISPLAY_LEVELS + ["Total"]
        data_a = [header_a]
        
        for cat in sorted_cats:
            row = [cat] # 第一列是类型
            row_total = 0
            for lvl in DISPLAY_LEVELS:
                # 注意：这里取值还是从 stats_cat_lev[lvl][cat] 取
                cnt = stats_cat_lev[lvl][cat]
                row.append(str(cnt) if cnt > 0 else "0")
                row_total += cnt
            row.append(str(row_total))
            data_a.append(row)
        
        # 动态列宽设置
        # 第一列类型给宽一点 (2.0 inch), 后面数字列给窄一点
        col_w_a = [2.0*inch] + [1.0*inch]*len(DISPLAY_LEVELS) + [0.8*inch]
        t_a = Table(data_a, hAlign='LEFT', colWidths=col_w_a, rowHeights=25)
        t_a.setStyle(self._get_summary_table_style())
        elements.append(t_a)
        elements.append(Spacer(1, 25))


        # ==========================================
        # 2. 统计表 B: Elevation & Severity (保持你截图的样式)
        # 行：Elevation
        # 列：Severity
        # ==========================================
        elements.append(Paragraph("Summary by Elevation & Severity:", self.styles["font_section"]))
        
        # 表头: [Elevation] + [Minor, Moderate, Major] + [Total]
        header_b = ["Elevation"] + DISPLAY_LEVELS + ["Total"]
        data_b = [header_b]

        for ele in sorted_eles:
            row = [ele] # 第一列是立面
            row_total = 0
            for lvl in DISPLAY_LEVELS:
                cnt = stats_ele_lev[ele][lvl]
                row.append(str(cnt) if cnt > 0 else "0")
                row_total += cnt
            row.append(str(row_total))
            data_b.append(row)
        
        # 列宽设置
        col_w_b = [2.0*inch] + [1.0*inch]*len(DISPLAY_LEVELS) + [0.8*inch]
        t_b = Table(data_b, hAlign='LEFT', colWidths=col_w_b, rowHeights=25)
        t_b.setStyle(self._get_summary_table_style())
        elements.append(t_b)
        
        elements.append(PageBreak())

    def _get_summary_table_style(self):
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ])
    def generate_flowables(self, df_record):
        elems = []
        
        first_row = df_record.iloc[0]
        fname = Path(first_row['Path']).name
        
        # 1. 标题 (灰色背景条)
        title_style = ParagraphStyle(
            'CompactTitle', 
            parent=self.styles['Heading2'], 
            backColor=colors.lightgrey, 
            borderPadding=5,
            spaceAfter=5
        )
        elems.append(Paragraph(f"File: {fname}", title_style))

        # 2. 全景大图 (关键修复：同时限制宽高)
        vis_path = first_row['VisPath']
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            
            # --- 【核心修改点】 ---
            # 定义最大允许尺寸 (根据报错信息 frame 636x456 设定)
            # 留出一点余量给标题和页脚
            limit_w = 8.5 * inch  # 限制宽度 (约 612 pts)
            limit_h = 5.0 * inch  # 限制高度 (约 360 pts，确保不会顶出页面)

            # 获取原始宽高比
            img_w, img_h = vis_img.imageWidth, vis_img.imageHeight
            aspect = img_h / img_w if img_w > 0 else 1.0

            # 1. 先尝试按最大宽度缩放
            draw_w = limit_w
            draw_h = draw_w * aspect

            # 2. 如果算出的高度超标，则改用高度限制来反算宽度
            if draw_h > limit_h:
                draw_h = limit_h
                draw_w = draw_h / aspect
            
            # 应用计算后的尺寸
            vis_img.drawWidth = draw_w
            vis_img.drawHeight = draw_h
            
            elems.append(vis_img)
            elems.append(Spacer(1, 5))

        # 3. 表格数据 (保持不变)
        headers = ["No.", "Defect ID", "Location", "Floor", "Dimension\n(L x W = Area)", "Severity", "Comment", "Action", "Image"]
        table_data = [headers]
        row_heights = [30]

        for idx, row in df_record.iterrows():
            # 尺寸
            if row['W_cm'] != 'N/A':
                dim_str = f"{row['H_cm']}L * {row['W_cm']}W\n= {row['Area_cm2']} cm²"
            else:
                dim_str = f"{row['H_pix']}px L * {row['W_pix']}px W\n= {row['Area_pix']} px²"

            # 裁剪图处理 (同样限制小图的大小)
            crop_path = row['CropPath']
            img_cell = ""
            row_h = 40 

            if os.path.exists(crop_path):
                img = RLImage(crop_path)
                
                # 限制小图尺寸
                target_w = 1.3 * inch 
                max_h = 1.8 * inch    
                
                aspect = img.imageHeight / img.imageWidth if img.imageWidth > 0 else 1.0
                
                # 双重限制计算
                calc_h = target_w * aspect
                if calc_h > max_h:
                    img.drawHeight = max_h
                    img.drawWidth = max_h / aspect
                else:
                    img.drawWidth = target_w
                    img.drawHeight = calc_h
                
                img_cell = img
                row_h = img.drawHeight + 6 

            # 等级
            lvl = row['Level']
            display_lvl = 'Minor' if lvl == 'Slight' else ('Major' if lvl == 'Serious' else lvl)

            table_row = [
                str(idx + 1),
                f"DF{idx+1}", 
                f"{row.get('view', '')}\n{row.get('orientation', '')}",
                str(row.get('floor', '-')),
                dim_str,
                display_lvl,
                "", 
                row['Action'],
                img_cell
            ]
            table_data.append(table_row)
            row_heights.append(row_h)

        # 定义列宽
        col_widths = [
            0.4*inch, 0.7*inch, 1.4*inch, 0.6*inch, 
            1.5*inch, 0.8*inch, 1.0*inch, 0.8*inch, 
            1.5*inch 
        ]
        
        t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
        
        style_list = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]
        t.setStyle(TableStyle(style_list))
        
        elems.append(t)
        elems.append(Spacer(1, 15))

        return elems