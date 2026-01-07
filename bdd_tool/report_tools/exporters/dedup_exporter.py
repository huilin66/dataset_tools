from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Image as RLImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.units import inch
from exporters.pdf_styles import PDFExporterCompact
import os
from reportlab.lib.styles import ParagraphStyle

class PDFExporterDedup(PDFExporterCompact):
    
    def _add_summary_pages(self, elements, report_data):
        # 扩展摘要页，增加 Elevation 信息
        input_info = report_data['input']
        output_info = report_data['output']
        elevation = output_info.get('elevation', '')
        
        # 标题
        elements.append(Paragraph(f"<b>View Report: {input_info['type']}</b>", self.styles["font_title"]))
        if elevation:
             elements.append(Paragraph(f"Elevation Orientation: {elevation}", self.styles["font_section"]))
        elements.append(Spacer(1, 20))
        
        # 调用父类逻辑生成统计表 (父类方法通常比较通用)
        super()._add_summary_pages(elements, report_data)

    def generate_flowables(self, df_view):
        elems = []
        if df_view.empty: return elems

        # 按 ID 和 楼层 排序
        df_sorted = df_view.sort_values(by=['ID', 'Floor'])
        
        # 【关键】更新表头，包含物理信息
        headers = ["ID", "Floor", "Height", "Size", "Defect Type", "Severity", "Image", "Action", "Crop"]
        table_data = [headers]
        row_heights = [30] 
        
        style_list = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]
        # 定义一个用于表格内自动换行的样式
        cell_style = ParagraphStyle(
            'CellStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10, # 行间距
            alignment=1 # 居中
        )

        last_id = -1
        for idx, row in df_sorted.iterrows():
            current_id = row['ID']
            
            # Crop 图处理
            crop_path = row['CropPath']
            img_cell = ""
            row_h = 45 
            if os.path.exists(crop_path):
                img = RLImage(crop_path)
                target_w = 1.0 * inch
                max_h = 1.5 * inch
                aspect = img.imageHeight / img.imageWidth if img.imageWidth > 0 else 1.0
                calc_h = target_w * aspect
                
                if calc_h > max_h:
                    img.drawHeight = max_h
                    img.drawWidth = max_h / aspect
                else:
                    img.drawWidth = target_w
                    img.drawHeight = calc_h
                img_cell = img
                row_h  = max(50, img.drawHeight + 10)

            # 使用 Paragraph 处理长文本 (Image Path)
            img_name = os.path.basename(row['VisPath'])
            img_para = Paragraph(img_name, cell_style)
            
            # 使用 Paragraph 处理多行文本 (Size)
            size_str = row.get('Real_Size', '-').replace('\n', '<br/>')
            size_para = Paragraph(size_str, cell_style)

            # 【关键】数据列对应
            table_row = [
                f"{row['ID']}",
                str(row.get('Floor', '-')),        # 从 json 获取的 Floor
                str(row.get('World_Z', '-')) + "m",# 从 json 获取的 Z
                size_para,       # 使用 Paragraph
                row['Category'],
                row['Level'],
                img_para,        # 使用 Paragraph
                row['Action'],
                img_cell
            ]
            
            table_data.append(table_row)
            row_heights.append(row_h)
            
            # ID 分隔线
            if last_id != -1 and current_id != last_id:
                style_list.append(('LINEBELOW', (0, len(table_data)-2), (-1, len(table_data)-2), 1.5, colors.black))
            last_id = current_id

        # 调整列宽 (Total ~ 7.5 inch)
        col_widths = [
            0.4*inch, # ID
            0.5*inch, # Floor
            0.6*inch, # Abs Z
            1.0*inch, # Size (Wider)
            0.9*inch, # Type
            0.7*inch, # Level
            1.5*inch, # Image File (Wider)
            0.7*inch, # Action
            1.2*inch  # Crop
        ]

        t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
        t.setStyle(TableStyle(style_list))
        
        elems.append(Paragraph(f"Detailed Defect List (Sorted by ID)", self.styles["font_section"]))
        elems.append(Spacer(1, 10))
        elems.append(t)
        
        return elems