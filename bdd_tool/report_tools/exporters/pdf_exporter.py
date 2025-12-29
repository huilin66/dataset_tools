# exporters/pdf_exporter.py
import time
import os
from pathlib import Path
from tqdm import tqdm

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch
from config import FONT_PATH_REGULAR, FONT_PATH_BOLD

class PDFExporter:
    def __init__(self):
        self._init_styles()

    def _init_styles(self):
        try:
            # 使用配置中的路径
            pdfmetrics.registerFont(TTFont("TimesNewRoman", FONT_PATH_REGULAR))
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", FONT_PATH_BOLD))
            self.FONT_REGULAR, self.FONT_BOLD = "TimesNewRoman", "TimesNewRoman-Bold"
        except:
            print("Warning: Custom font not found, using default.")
            self.FONT_REGULAR, self.FONT_BOLD = "Helvetica", "Helvetica-Bold"

        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="font_title", fontName=self.FONT_BOLD, fontSize=22, alignment=1, leading=33))
        self.styles.add(ParagraphStyle(name="font_section", fontName=self.FONT_BOLD, fontSize=20, leading=30))
        self.styles.add(ParagraphStyle(name="font_text", fontName=self.FONT_REGULAR, fontSize=16, leading=24))

        self.threeline_table = TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 2, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 14),
            ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
        ])

        self.blank_table = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 16),
        ])
        
        self.new_threeline_table = TableStyle(self.threeline_table.getCommands())

    def export(self, report_data, save_path):
        print(f"[{time.strftime('%H:%M:%S')}] PDF Generation started...")
        input_info = report_data['input']
        output_info = report_data['output']
        records_df_list = report_data['records']

        doc = SimpleDocTemplate(save_path, pagesize=letter)
        elements = []
        
        # --- Summary Section ---
        elements.append(Paragraph("<b>AI-Detection Result Report</b>", self.styles["font_title"]))
        elements.append(Spacer(1, 30))

        elements.append(Paragraph("Input Information:", self.styles["font_section"]))
        shape_str = f"{input_info['shape'][0]}~{input_info['shape'][1]}, {input_info['shape'][2]}~{input_info['shape'][3]}"
        data_input = [
            ["Type of Data:", input_info['type'].title()],
            ["Number of Images:", input_info['number']],
            ["Shape Range (W, H):", shape_str]
        ]
        t_input = Table(data_input, hAlign='LEFT', rowHeights=25)
        t_input.setStyle(self.blank_table)
        elements.append(t_input)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Detection Summary:", self.styles["font_section"]))
        data_output_summary = [
            ["Model Used:", output_info['model']],
            ["Images with Defects:", output_info['defects']],
            ["Images without Defects:", output_info['no defects']],
        ]
        t_summary = Table(data_output_summary, hAlign='LEFT', rowHeights=25)
        t_summary.setStyle(self.blank_table)
        elements.append(t_summary)
        elements.append(Spacer(1, 10))

        data_output_defects = [["Category", "Count"]]
        for k, v in output_info['defects sta'].items():
            data_output_defects.append([k.title(), v])
        t_stats = Table(data_output_defects, hAlign='CENTER', rowHeights=25)
        t_stats.setStyle(self.threeline_table)
        elements.append(t_stats)
        elements.append(PageBreak())

        # --- Details Section ---
        elements.append(Paragraph("Detailed Information:", self.styles["font_section"]))
        elements.append(Spacer(1, 10))

        data_records = []
        rows_h = []

        def add_row(row_data, height):
            data_records.append(row_data)
            rows_h.append(height if height is not None else 20)

        # 构建表格内容
        for df_record in tqdm(records_df_list, desc="Building PDF Content"):
            if df_record.empty: continue
            
            first_row = df_record.iloc[0]
            vis_path = first_row['VisPath']
            file_name = Path(first_row['Path']).name
            
            if os.path.exists(vis_path):
                vis_img = RLImage(vis_path)
                aspect = vis_img.drawHeight / vis_img.drawWidth if vis_img.drawWidth > 0 else 1.0
                vis_img.drawWidth = 5 * inch
                vis_img.drawHeight = 5 * inch * aspect
                img_h = vis_img.drawHeight * 1.05
            else:
                vis_img = "Image Not Found"
                img_h = 25 

            add_row(["FileName", file_name], 25)
            add_row([vis_img, ''], img_h)
            add_row(['Number of Defects', str(len(df_record))], 25)

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

        if data_records:
            if len(data_records) != len(rows_h):
                min_len = min(len(data_records), len(rows_h))
                data_records = data_records[:min_len]
                rows_h = rows_h[:min_len]

            t_records = Table(data_records, hAlign='CENTER', colWidths=[2*inch, 4*inch], rowHeights=rows_h)
            
            final_style = TableStyle(self.new_threeline_table.getCommands())
            for i, row_data in enumerate(data_records):
                if row_data[0] == 'FileName':
                    final_style.add('SPAN', (0, i), (-1, i))
                    final_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
            
            t_records.setStyle(final_style)
            elements.append(t_records)
        else:
            elements.append(Paragraph("No defects detected.", self.styles["font_text"]))

        print("Writing PDF to disk...")
        try:
            doc.build(elements)
            print(f"[{time.strftime('%H:%M:%S')}] Success! Report saved to {save_path}")
        except Exception as e:
            print(f"[ERROR] PDF Build Failed: {e}")