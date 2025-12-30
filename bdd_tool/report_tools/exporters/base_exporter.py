# exporters/base_exporter.py
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import config

class BasePDFExporter:
    def __init__(self):
        self._init_fonts()
        self._init_styles()

    def _init_fonts(self):
        try:
            pdfmetrics.registerFont(TTFont("TimesNewRoman", config.FONT_PATH_REGULAR))
            pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", config.FONT_PATH_BOLD))
            self.FONT_REGULAR, self.FONT_BOLD = "TimesNewRoman", "TimesNewRoman-Bold"
        except:
            self.FONT_REGULAR, self.FONT_BOLD = "Helvetica", "Helvetica-Bold"

    def _init_styles(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="font_title", fontName=self.FONT_BOLD, fontSize=22, alignment=1, leading=33))
        self.styles.add(ParagraphStyle(name="font_section", fontName=self.FONT_BOLD, fontSize=20, leading=30))
        self.styles.add(ParagraphStyle(name="font_text", fontName=self.FONT_REGULAR, fontSize=16, leading=24))

        self.table_style_common = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 14),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        
        # 定义三种表格样式供子类调用
        self.style_blank = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR)])
        self.style_threeline = TableStyle([("LINEABOVE", (0, 0), (-1, 0), 2, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black), ("VALIGN", (0, 0), (-1, -1), 'MIDDLE')])

    def generate_row_content(self, df_record):
        """
        【抽象方法】这是子类必须实现的核心差异方法。
        输入：单张图片的 DataFrame
        输出：(data_rows, row_heights)
        """
        raise NotImplementedError("Subclasses must implement generate_row_content")

    def export(self, report_data, save_path):
        print(f"[{time.strftime('%H:%M:%S')}] PDF Generation started ({self.__class__.__name__})...")

        # --- 【修改点】动态页面大小 ---
        # 默认使用 Letter 纵向，如果子类定义了 pagesize 属性（如横向），则使用子类的
        target_pagesize = getattr(self, 'pagesize', letter)
        doc = SimpleDocTemplate(save_path, pagesize=target_pagesize)
        
        elements = []

        # --- 1. 通用表头与摘要 ---
        self._add_summary_pages(elements, report_data)

        # --- 2. 详细内容 ---
        elements.append(Paragraph("Detailed Information:", self.styles["font_section"]))
        elements.append(Spacer(1, 10))

        records_df_list = report_data['records']

        # ==========================================
        # 【修改核心】：分支处理流式布局 vs 表格布局
        # ==========================================
        
        # 模式 A: 流式布局 (Style 3 - 解决超长表格跨页问题)
        if hasattr(self, 'generate_flowables'):
            from tqdm import tqdm
            print("Generating flowables (Stream Mode)...")
            
            for df_record in tqdm(records_df_list, desc="Processing Images"):
                if df_record.empty: continue
                # 直接获取元素列表（标题、图、表...）并加入主流程
                record_elements = self.generate_flowables(df_record)
                elements.extend(record_elements)
                # 每张图片处理完后强制分页，保持报告整洁（可选，也可改为 Spacer）
                elements.append(PageBreak())

        # 模式 B: 统一大表格布局 (Style 0, 1, 2 - 保持原有逻辑)
        else:
            all_data_rows = []
            all_row_heights = []

            from tqdm import tqdm
            for df_record in tqdm(records_df_list, desc="Building Table Rows"):
                if df_record.empty: continue
                
                # 调用子类的方法获取行数据
                rows, heights = self.generate_row_content(df_record)
                all_data_rows.extend(rows)
                all_row_heights.extend(heights)

            # --- 3. 构建并保存表格 ---
            if all_data_rows:
                # 简单的长度保护
                min_len = min(len(all_data_rows), len(all_row_heights))
                all_data_rows = all_data_rows[:min_len]
                all_row_heights = all_row_heights[:min_len]

                from reportlab.lib.units import inch
                # 基础布局还是 2 列，但我们会通过 Span 让 Style 3 变宽
                t = Table(all_data_rows, hAlign='CENTER', colWidths=[2.5*inch, 4.5*inch], rowHeights=all_row_heights)
                
                final_style = TableStyle(self.table_style_common.getCommands())
                
                for i, row in enumerate(all_data_rows):
                    # 1. 文件名行 (灰色背景 + 跨列)
                    if row[0] == 'FileName':
                        final_style.add('SPAN', (0, i), (-1, i))
                        final_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
                    
                    # 2. 【新增】全宽行支持 (用于 Style 3 的横向表格)
                    # 如果行的第一个元素是 'FullWidth'，我们只显示第二个元素的内容，并让它跨列
                    elif row[0] == 'FullWidth':
                        final_style.add('SPAN', (0, i), (-1, i))
                        # 去掉中间的分隔线，只保留外框（可选）
                        # final_style.add('BOX', (0, i), (-1, i), 1, colors.black)

                t.setStyle(final_style)
                elements.append(t)
            else:
                elements.append(Paragraph("No defects detected.", self.styles["font_text"]))

        try:
            # ==========================================
            # 【修改点】：在 doc.build 前增加提示信息
            # ==========================================
            print(f"[{time.strftime('%H:%M:%S')}] Compiling PDF and writing to disk... Please wait.")
            
            doc.build(elements)
            
            print(f"[{time.strftime('%H:%M:%S')}] Success! Report saved to {save_path}")
            # ==========================================
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _add_summary_pages(self, elements, report_data):
        """生成摘要页的通用逻辑"""
        input_info = report_data['input']
        output_info = report_data['output']
        
        # 1. 标题
        elements.append(Paragraph("<b>AI-Detection Result Report</b>", self.styles["font_title"]))
        elements.append(Spacer(1, 30))

        # 2. Input Info 表格
        elements.append(Paragraph("Input Information:", self.styles["font_section"]))
        shape_str = f"{input_info['shape'][0]}~{input_info['shape'][1]}, {input_info['shape'][2]}~{input_info['shape'][3]}"
        data_input = [
            ["Type of Data:", input_info['type'].title()],
            ["Number of Images:", str(input_info['number'])],
            ["Shape Range:", shape_str]
        ]
        t_input = Table(data_input, hAlign='LEFT', rowHeights=25, colWidths=[200, 300])
        t_input.setStyle(self.style_blank)
        elements.append(t_input)
        elements.append(Spacer(1, 20))

        # 3. Detection Summary 表格
        elements.append(Paragraph("Detection Summary:", self.styles["font_section"]))
        data_output = [
            ["Model Used:", output_info['model']],
            ["With Defects:", str(output_info['defects'])],
            ["No Defects:", str(output_info['no defects'])],
        ]
        t_output = Table(data_output, hAlign='LEFT', rowHeights=25, colWidths=[200, 300])
        t_output.setStyle(self.style_blank)
        elements.append(t_output)
        elements.append(Spacer(1, 20))
        
        # 4. Statistics 表格 (类别统计)
        if output_info.get('defects sta'):
            elements.append(Paragraph("Defect Statistics:", self.styles["font_section"]))
            data_stats = [["Category", "Count"]] # 表头
            for cat, count in output_info['defects sta'].items():
                data_stats.append([cat, str(count)])
            
            t_stats = Table(data_stats, hAlign='LEFT', rowHeights=25, colWidths=[200, 100])
            t_stats.setStyle(self.style_threeline) # 使用三线表样式
            elements.append(t_stats)
            elements.append(Spacer(1, 20))

        elements.append(PageBreak())