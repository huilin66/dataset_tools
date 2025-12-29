# exporters/pdf_styles.py
import os
from pathlib import Path
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Table, Spacer, Paragraph
from reportlab.lib.units import inch
from .base_exporter import BasePDFExporter

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

# exporters/pdf_styles.py

# ... (保持 imports 和 PDFExporterBasic, PDFExporterDetailed 不变) ...

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