# exporters/base_exporter.py
from collections import defaultdict
import concurrent.futures
from copy import deepcopy
import os
from pathlib import Path
import re
import time
import uuid

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, portrait, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    BaseDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Frame,
    PageTemplate,
    NextPageTemplate,
)
from reportlab.platypus import Flowable
from reportlab.platypus.tableofcontents import TableOfContents
from tqdm import tqdm
import config

class TOCEntryNotifier(Flowable):
    """
    一个看不见的组件，专门负责在渲染时把当前的页码告诉 TOC 对象。
    替代之前的 Macro 字符串，解决作用域报错问题。
    """
    def __init__(self, toc_ref, level, text, key):
        super().__init__()
        self.toc_ref = toc_ref  # 直接持有 toc 对象的引用
        self.level = level
        self.text = text
        self.key = key
        # 设置宽高为0，不可见
        self.width = 0
        self.height = 0

    def draw(self):
        # 这个 draw 方法会在生成 PDF 的瞬间被调用
        # self.canv.getPageNumber() 获取当前页码
        # 将 (层级, 标题文本, 页码, 跳转Key) 存入 toc
        self.toc_ref.addEntry(self.level, self.text, self.canv.getPageNumber(), key=self.key)

class PDFBookmark(Flowable):
    """自定义 Flowable，用于在 PDF 中插入书签节点"""
    def __init__(self, title, level=0):
        super().__init__()
        self.title = title
        self.level = level
        # 生成唯一 key，防止书签冲突
        self.key = str(uuid.uuid4())
        # 设置宽高为0，不占用可见空间
        self.width = 0
        self.height = 0

    def draw(self):
        # 1. 在当前页标记位置
        self.canv.bookmarkPage(self.key)
        # 2. 添加大纲条目 (closed=True 默认折叠，False 默认展开)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=True)


class BasePDFExporter:
    def __init__(self, logo_left, logo_right, target_cls_names=None, max_workers=1):
        self._init_fonts()
        self._init_styles()
        self.logo_left = logo_left
        self.logo_right = logo_right
        self.target_cls_names = target_cls_names
        self.app_name = config.APP_NAME
        self.project_name = config.PROJECT_NAME
        self.originze_name = config.ORIGINZE_NAME
        self.cover_image = config.COVER_IMAGE
        self.app_description = config.APP_DESCEPTION
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

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
        self.styles.add(ParagraphStyle(name="font_subsection", fontName=self.FONT_REGULAR, fontSize=18, leading=28))
        self.styles.add(ParagraphStyle(name="font_text", fontName=self.FONT_REGULAR, fontSize=16, leading=24))
        # --- [新增] 封面专用样式 ---
        # 1. 主标题 (BD-Detection) - 深蓝灰色，字号很大，加粗
        self.styles.add(ParagraphStyle(
            name="CoverTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=42,
            leading=50,
            textColor=HexColor("#2C3E50"), # 深蓝灰色
            spaceBefore=20,
            spaceAfter=10
        ))

        # 2. 副标题 (AI-based...) - 蓝色，中等字号
        self.styles.add(ParagraphStyle(
            name="CoverSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=18,
            leading=24,
            textColor=HexColor("#3498DB"), # 亮蓝色
            spaceAfter=30
        ))

        # 3. 底部版权文字 - 灰色，小字号
        self.styles.add(ParagraphStyle(
            name="CoverCopyright",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.gray,
            alignment=1 # 居中
        ))

        self.table_style_common = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 14),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        
        self.style_blank = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR)])
    
    def _add_cover_page(self, elements):
        """
        生成首页 (Cover Page)
        """
        # 1. 顶部留白 (给 Logo 留位置)
        # 如果你希望 Logo 是通过 flowable 添加的，可以在这里加 Image
        # 如果 Logo 是通过 canvas 画在固定位置的（在 _draw_cover_layout 中），这里只需留白
        elements.append(Spacer(1, 1.0 * inch)) 

        # 2. 中间大图 (Main Image)
        if self.cover_image and os.path.exists(self.cover_image):
            # 创建图片并自适应宽度
            img = self._create_rl_image(self.cover_image, 9.0 * inch, 6.0 * inch)
            if img:
                # 图片居中 (使用 Table 包裹或者设置 hAlign)
                img.hAlign = 'CENTER'
                elements.append(img)
                elements.append(Spacer(1, 0.5 * inch))
        else:
            # 如果没有图片，留出一大块空白占位
            elements.append(Spacer(1, 4.0 * inch))

        # 3. 软件名称 (BD-Detection)
        elements.append(Paragraph(self.app_name, self.styles["CoverTitle"]))

        # 4. 解释/副标题 (AI-based...)
        # 使用 <u> 标签可以加下划线，但图片中似乎是链接样式，这里用颜色区分即可
        elements.append(Paragraph(self.app_description, self.styles["CoverSubtitle"]))

        # 5. 底部推挤 (Spacer)
        # 为了让版权信息沉底，我们加一个大的 Spacer，或者依靠 Canvas 在页脚绘制
        # 建议：这里只负责内容，底部的 Copyright Logo 建议用 Canvas 绘制以保证位置绝对靠下
        elements.append(PageBreak())


    def _add_toc(self, elements):
        elements.append(Paragraph("Table of Contents", self.styles["font_title"]))
        elements.append(Spacer(1, 20)) # 1. 标题和目录内容之间留点空隙
        
        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(fontName=self.FONT_BOLD, fontSize=14, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceBefore=10, leading=20),
            ParagraphStyle(fontName=self.FONT_REGULAR, fontSize=12, name='TOCHeading2', leftIndent=40, firstLineIndent=-20, spaceBefore=5, leading=16),
        ]
        elements.append(self.toc)           # 3. 将 TOC 对象加入到 PDF 元素列表中
        
        elements.append(PageBreak())   # 4. 目录通常独占一页，所以结束后加分页符


    def _add_section_title(self, elements, text, level=0):
        """
        核心辅助函数：添加标题并注册到目录
        level 0 = 一级标题 (1. xxx)
        level 1 = 二级标题 (1.1 xxx)
        """
        # 1. 生成唯一的 key 用于书签跳转
        key = str(uuid.uuid4())
        
        # 2. 选择样式
        style = self.styles["font_section"] if level == 0 else self.styles["font_subsection"]
        
        # 3. 添加带书签锚点的标题段落
        # 格式: <a name="KEY"/>Title Text
        p = Paragraph(f'<a name="{key}"/>{text}', style)
        elements.append(p)

        elements.append(PDFBookmark(text, level))
        
        # 2. 添加通知器 (使用 TOCEntryNotifier)
        # 只要 self.toc 已经被初始化，这里就能直接传进去
        if hasattr(self, 'toc'):
            notifier = TOCEntryNotifier(self.toc, level, text, key)
            elements.append(notifier)
        else:
            print('toc not found!')

    # --- 通用工具方法 (新增) ---
    def _create_rl_image(self, path, max_w, max_h=None):
        """统一处理图片加载、按比例缩放逻辑"""
        if not path or not os.path.exists(path):
            return None
        
        try:
            img = RLImage(path)
            img_w, img_h = img.imageWidth, img.imageHeight
            aspect = img_h / img_w if img_w > 0 else 1.0
            
            draw_w = max_w
            draw_h = draw_w * aspect
            
            # 如果指定了最大高度且超标，则基于高度反算宽度
            if max_h and draw_h > max_h:
                draw_h = max_h
                draw_w = draw_h / aspect
            
            img.drawWidth = draw_w
            img.drawHeight = draw_h
            return img
        except Exception:
            return None
    def _draw_logo_on_canvas(self, canvas, img_path, x, y, height):
        """辅助函数：在 Canvas 上画图 (自动保持比例)"""
        if img_path and os.path.exists(img_path):
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                aspect = iw / ih if ih > 0 else 1
                width = height * aspect
                canvas.drawImage(img_path, x, y, width=width, height=height, mask='auto')
            except:
                pass
    def _draw_cover(self, canvas, doc):
        canvas.saveState()
        w, h = doc.pagesize
        # ==========================
        # 场景 A: 首页 (Cover Page)
        # 1. 绘制顶部的 Logo (如果需要)
        # 假设 self.logo_left 是你的 logo 路径
        if hasattr(self, 'logo_left') and self.logo_left:
            self._draw_logo_on_canvas(canvas, self.logo_left, x=50, y=h-80, height=40)

        # 绘制文字
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        copyright_text = "Copyright © SCRI. All Rights Reserved."
        # 居中显示在底部 30pt 处
        canvas.drawCentredString(w/2, 30, copyright_text)
        
        canvas.restoreState()
        return  # <--- 重要！首页画完后直接退出，不画后面的红线

    def _draw_custom_header_footer(self, canvas, doc):
        """
        绘制带有红线、Project ID 和页码的页眉页脚
        """
        canvas.saveState()

        # --- 获取页面尺寸和边距 ---
        w, h = canvas._pagesize

        # 假设左右页边距是 0.75 inch (约 54 points)
        margin_x = 0.75 * inch 
        header_y = h - 0.75 * inch  # 页眉红线高度
        footer_y = 0.75 * inch      # 页脚红线高度
        
        # 定义颜色 (图片中的暗红色)
        theme_color = HexColor("#A03030") 
        
        # ====================
        # 1. 绘制页眉 (Header)
        # ====================
        # A. 画红线
        canvas.setStrokeColor(theme_color)
        canvas.setLineWidth(1)
        canvas.line(margin_x, header_y, w - margin_x, header_y)
        
        # B. 写文字 (Project ID)
        canvas.setFont(self.FONT_REGULAR, 9)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(w/2, header_y + 5, self.project_name)
        
        # (可选) 如果你还需要左上角的 Logo，可以在这里调用之前的逻辑
        self._draw_header_logos(canvas, doc) 

        # ====================
        # 2. 绘制页脚 (Footer)
        # ====================
        # A. 画红线
        canvas.setStrokeColor(theme_color)
        canvas.setLineWidth(1)
        canvas.line(margin_x, footer_y, w - margin_x, footer_y)
        
        # B. 左侧文字 (Powered by...)
        footer_text = f"Powered by {self.originze_name}"
        canvas.setFont(self.FONT_REGULAR, 9)
        canvas.setFillColor(colors.black)
        canvas.drawString(margin_x, footer_y - 12, footer_text)
        
        # C. 中间页码 (- 1 -)
        page_num_text = f"- {doc.page} -"
        canvas.drawCentredString(w / 2, footer_y - 12, page_num_text)
        
        canvas.restoreState()

    def _draw_header_logos(self, canvas, doc):
        """回调函数：绘制页眉 Logo"""
        canvas.saveState()
        page_w, page_h = canvas._pagesize
        
        target_h = 0.45 * inch
        margin = 20
        
        # 绘制 helper
        def draw_logo(path, is_right_align=False):
            if path and os.path.exists(path):
                try:
                    img = ImageReader(path)
                    iw, ih = img.getSize()
                    aspect = iw / ih if ih > 0 else 1
                    draw_h = target_h
                    draw_w = draw_h * aspect
                    
                    y = page_h - margin - draw_h
                    if is_right_align:
                        x = page_w - margin - draw_w
                    else:
                        x = margin
                    canvas.drawImage(path, x, y, width=draw_w, height=draw_h, mask='auto')
                except Exception as e:
                    print(f"Error drawing logo {path}: {e}")

        draw_logo(getattr(self, 'logo_left', None), is_right_align=False)
        draw_logo(getattr(self, 'logo_right', None), is_right_align=True)
        canvas.restoreState()

    def generate_row_content(self, df_record):
        raise NotImplementedError("Subclasses must implement generate_row_content")

    def export(self, report_data, save_path):
        print(f"[{time.strftime('%H:%M:%S')}] PDF Task queued: {os.path.basename(save_path)}")
        future = self.executor.submit(self._execute_export_sync, report_data, save_path)
        future.add_done_callback(lambda f: self._on_export_complete(f, save_path))
        return future

    def _on_export_complete(self, future, save_path):
        try:
            future.result()
            print(f"Finished: report exported to {save_path}\n")
        except Exception as e:
            print(f"!!! Error exporting {save_path}: {e}")

    def _execute_export_sync(self, report_data, save_path):
        print(f"[{time.strftime('%H:%M:%S')}] PDF Generation started in background ({self.__class__.__name__})...\n")
        
        # 1. 使用 BaseDocTemplate，初始化为纵向 A4
        doc = BaseDocTemplate(save_path, pagesize=portrait(A4), topMargin=1.0*inch, bottomMargin=1.0*inch)
        
        # --- 定义 Frame (内容区域) ---
        # 纵向 Frame (用于封面、目录、摘要)
        w_p, h_p = portrait(A4)
        frame_p = Frame(
            doc.leftMargin, doc.bottomMargin, 
            w_p - doc.leftMargin - doc.rightMargin, 
            h_p - doc.topMargin - doc.bottomMargin, 
            id='portrait_frame'
        )
        
        # 横向 Frame (用于详细大表)
        w_l, h_l = landscape(A4)
        frame_l = Frame(
            doc.leftMargin, doc.bottomMargin, 
            w_l - doc.leftMargin - doc.rightMargin, 
            h_l - doc.topMargin - doc.bottomMargin, 
            id='landscape_frame'
        )

        # --- 定义三个专属模板 ---
        
        # 模板 A: 封面 (Cover)
        # 逻辑：使用 _draw_cover 绘制，分页后自动跳到 'Portrait' 模板
        cover_template = PageTemplate(
            id='Cover',
            frames=frame_p,
            onPage=self._draw_cover, 
            pagesize=portrait(A4),
            autoNextPageTemplate='Portrait'  # [关键] 封面后强制变纵向
        )

        # 模板 B: 普通纵向 (Portrait)
        # 逻辑：使用 _draw_custom_header_footer，分页后保持 'Portrait'
        portrait_template = PageTemplate(
            id='Portrait',  # 改个名字，比 'Normal' 更清晰
            frames=frame_p,
            onPage=self._draw_custom_header_footer,
            pagesize=portrait(A4),
            autoNextPageTemplate='Portrait'  # [关键] 保持纵向
        )

        # 模板 C: 横向 (Landscape)
        # 逻辑：使用 _draw_custom_header_footer，分页后保持 'Landscape'
        landscape_template = PageTemplate(
            id='Landscape',
            frames=frame_l,
            onPage=self._draw_custom_header_footer,
            pagesize=landscape(A4),
            autoNextPageTemplate='Landscape' # [关键] 保持横向，解决"只有第一页横向"的问题
        )
        
        # 加入所有模板 (Cover 在第一个，所以是默认起始页)
        doc.addPageTemplates([cover_template, portrait_template, landscape_template])

        elements = []

        # --- 1. 封面 (Cover Template) ---
        self._add_cover_page(elements)
        # _add_cover_page 里的 PageBreak 会触发切换到 'Portrait'

        # --- 2. 目录和摘要 (Portrait Template) ---
        self._add_toc(elements)
        self._add_project_summary_pages(elements, report_data)
        
        # 注意：请确保 _add_result_summary_pages 内部最后没有 PageBreak
        self._add_result_summary_pages(elements, report_data) 

        # --- 3. 切换到横向 (Landscape Template) ---
        # 显式指令：下一页开始使用 Landscape
        elements.append(NextPageTemplate('Landscape'))
        elements.append(PageBreak()) # [关键] 必须在这里强制分页，才能让新页面应用 Landscape

        # --- 4. 详细内容 (Landscape Template) ---
        self._add_section_title(elements, "Detailed Information", level=0)
        elements.append(Spacer(1, 10))

        records_df_list = report_data['records']
        
        if hasattr(self, 'generate_flowables'):
            # 流式布局 (Compact / Context)
            for df_record in records_df_list:
                if df_record.empty: continue
                elements.extend(self.generate_flowables(df_record))
                elements.append(PageBreak()) 
        else:
            # 大表格布局
            self._render_big_table_mode(elements, records_df_list)

        elements.append(NextPageTemplate('Portrait'))
        elements.append(PageBreak())

        print(f"[{time.strftime('%H:%M:%S')}] Compiling PDF {os.path.basename(save_path)}...")
        try:
            # 使用 multiBuild，不需要再传 onPage 参数，因为已经在模板里绑定了
            doc.multiBuild(elements)
            print(f"[{time.strftime('%H:%M:%S')}] Success! Saved to {save_path}")
        except Exception as e:
            print(f"Error in thread: {e}")
            import traceback; traceback.print_exc()
            raise e


    def generate_flowables(self):
        raise NotImplementedError("Subclasses must implement generate_flowables")


    def _render_big_table_mode(self, elements, records_df_list):
        """处理 Basic/Detailed/Measurement 的大表格聚合逻辑"""
        all_data_rows = []
        all_row_heights = []

        for df_record in records_df_list:
            if df_record.empty: continue
            rows, heights = self.generate_row_content(df_record)
            all_data_rows.extend(rows)
            all_row_heights.extend(heights)

        if all_data_rows:
            # 简单的截断保护，防止两个列表长度不一致
            min_len = min(len(all_data_rows), len(all_row_heights))
            all_data_rows = all_data_rows[:min_len]
            all_row_heights = all_row_heights[:min_len]

            t = Table(all_data_rows, hAlign='CENTER', colWidths=[2.5*inch, 4.5*inch], rowHeights=all_row_heights)
            final_style = TableStyle(self.table_style_common.getCommands())
            
            for i, row in enumerate(all_data_rows):
                if row[0] == 'FileName':
                    final_style.add('SPAN', (0, i), (-1, i))
                    final_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
                elif row[0] == 'FullWidth':
                    final_style.add('SPAN', (0, i), (-1, i))

            t.setStyle(final_style)
            elements.append(t)
        else:
            elements.append(Paragraph("No defects detected.", self.styles["font_text"]))

    def _add_glossary_section(self, elements, title, description, items_dict):
        """辅助方法：生成定义列表表格（Defects 或 Levels）"""
        elements.append(Paragraph(title, self.styles["font_section"]))
        elements.append(Paragraph(description, self.styles["font_text"]))
        elements.append(Spacer(1, 10))

        table_rows = [[title.split(':')[0], "Description"]] # 简单的表头
        desc_style = ParagraphStyle('GlossaryDesc', parent=self.styles['Normal'], fontSize=10, leading=12)

        for k, v in items_dict.items():
            table_rows.append([Paragraph(k), Paragraph(str(v), desc_style)])

        # 根据页面方向调整列宽
        col_widths = [1.5 * inch, 4.5 * inch]
        # col_widths = [2.0 * inch, 7.0 * inch]

        t = Table(table_rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('padding', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

    def _add_project_summary_pages(self, elements, report_data):
        """生成汇总页"""
        input_info = report_data['input']
        output_info = report_data['output']

        # 1. Title & Basic Info
        self._add_section_title(elements, "Project Summary Report", level=0)
        self._add_section_title(elements, "Basic Information", level=1)
        elements.append(Spacer(1, 20))

        data_input = [
            ["Total Images:", str(input_info['number'])],
            ["Model:", output_info['model']],
            ["Total Detected Defects:", str(output_info['defects'])],
        ]
        t_input = Table(data_input, hAlign='LEFT', colWidths=[3*inch, 3*inch])
        t_input.setStyle(self.style_blank)
        elements.append(t_input)
        elements.append(Spacer(1, 20))

        # 2. Definitions Tables
        explanation_json = input_info.get('explanation_json', {})
        if explanation_json:
            elements.append(PDFBookmark("Defect Definitions", level=1))
            cats = deepcopy(explanation_json.get("Category", {}))
            cats_desc = cats.pop("Description", "")
            self._add_glossary_section(elements, "Defect Definitions:", cats_desc, cats)

            elements.append(PDFBookmark("Level Definitions", level=1))
            lvls = deepcopy(explanation_json.get("Levels", {}))
            lvls_desc = lvls.pop("Description", "")
            self._add_glossary_section(elements, "Level Definitions:", lvls_desc, lvls)
            
        elements.append(PageBreak())

        # 3. Views Map
        self._add_section_title(elements, "Views Direction Map", level=1)
        elements.append(Spacer(2, 20))
        img = self._create_rl_image(input_info.get('views_png_path'), 
                                    max_w=8.5*inch if isinstance(self, PDFExporterCompact) else 6.0*inch,
                                    max_h=5.0*inch)
        if img: elements.append(img)
        elements.append(PageBreak())


    def _add_result_summary_pages(self, elements, report_data):
        """统计数据处理逻辑"""
        self._add_section_title(elements, "Result Summary Report", level=0)

        records_list = report_data['records']

        stats_cat_lev = defaultdict(lambda: defaultdict(int))
        stats_view_lev = defaultdict(lambda: defaultdict(int))
        stats_ori_lev = defaultdict(lambda: defaultdict(int))
        stats_floor_lev = defaultdict(lambda: defaultdict(int))
        
        defined_cats = self.target_cls_names if self.target_cls_names else report_data.get('defined_categories', [])
        defined_floors = report_data.get('defined_floors', [])[::-1]
        
        all_categories, all_floors, all_views, all_orientations = set(defined_cats), set(defined_floors), set(), set()

        for df in records_list:
            if df.empty: continue
            for _, row in df.iterrows():
                cat = row['Category']
                level = row['Level']
                v = str(row.get('view', 'Unknown')).strip() or "Unknown"
                fl = str(row.get('floor', 'Unknown')).strip() or "Unknown"
                o = str(row.get('orientation', 'Unknown')).strip() or "Unknown"
                if v == 'nan': v = "Unknown"
                if fl == 'nan': fl = "Unknown"
                if o == 'nan': o = "Unknown"

                all_categories.add(cat)
                all_views.add(v)
                all_floors.add(fl)
                if o != "Unknown": all_orientations.add(o)

                stats_cat_lev[level][cat] += 1
                stats_view_lev[level][v] += 1
                stats_floor_lev[level][fl] += 1
                if o != "Unknown": stats_ori_lev[level][o] += 1

        # Sorting logic
        def natural_keys(text): return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(text))]
        
        if defined_cats:
            sorted_cats = defined_cats + sorted(list(all_categories - set(defined_cats)))
        else:
            sorted_cats = sorted(list(all_categories))
            
        sorted_views = sorted(list(all_views), key=natural_keys)
        dir_order = {'N':1, 'NE':2, 'E':3, 'SE':4, 'S':5, 'SW':6, 'W':7, 'NW':8}
        sorted_oris = sorted(list(all_orientations), key=lambda x: dir_order.get(x, 99))
        floor_order = {floor: index for index, floor in enumerate(defined_floors)}
        sorted_floors = sorted(list(all_floors), key=lambda floor: floor_order.get(floor, float('inf')))

        # Table Output Helper
        def add_stats_table(title, row_keys, stats_dict, col1_name):
            self._add_section_title(elements, title, level=1)
            if not row_keys: return

            header = [col1_name] + config.DISPLAY_LEVELS + ["Total"]
            data = [header]
            for key in row_keys:
                row = [key]
                total = 0
                for lvl in config.DISPLAY_LEVELS:
                    val = stats_dict[lvl][key]
                    row.append(str(val) if val > 0 else "0")
                    total += val
                row.append(str(total))
                data.append(row)
            
            col_w = [2.0*inch] + [1.0*inch]*len(config.DISPLAY_LEVELS) + [0.8*inch]
            t = Table(data, hAlign='LEFT', colWidths=col_w, rowHeights=25)
            
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
            ])
            t.setStyle(style)
            elements.append(t)
            elements.append(Spacer(1, 25))

        
        add_stats_table("Summary by Defect Type:", sorted_cats, stats_cat_lev, "Defect Type")
        add_stats_table("Summary by View:", sorted_views, stats_view_lev, "View ID")
        if sorted_oris: add_stats_table("Summary by Direction:", sorted_oris, stats_ori_lev, "Direction")
        add_stats_table("Summary by Floor:", sorted_floors, stats_floor_lev, "Floor")
        # elements.append(PageBreak())


# --- 基础报告系列 (Basic, Detailed, Measurement) ---

class PDFExporterBasic(BasePDFExporter):
    """ 样式 0: 基础报告 """
    def generate_row_content(self, df_record):
        data_rows, row_heights = [], []
        def add(row, h): data_rows.append(row); row_heights.append(h or 20)

        first_row = df_record.iloc[0]
        vis_img = self._create_rl_image(first_row['VisPath'], 5*inch)
        img_h = (vis_img.drawHeight * 1.05) if vis_img else 25
        
        add(["FileName", Path(first_row['Path']).name], 25)
        add([vis_img or "Missing", ''], img_h)
        add(['Defect Count', str(len(df_record))], 25)

        for idx, row in df_record.iterrows():
            crop_img = self._create_rl_image(row['CropPath'], 2*inch, 2*inch) or "Missing"
            add([f"ID: {row.get('ID', f'Defect {idx+1}')}", crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        return data_rows, row_heights


class PDFExporterDetailed(BasePDFExporter):
    """ 样式 1: 详细报告 """
    def generate_row_content(self, df_record):
        data_rows, row_heights = [], []
        def add(row, h): data_rows.append(row); row_heights.append(h or 20)

        first_row = df_record.iloc[0]
        vis_img = self._create_rl_image(first_row['VisPath'], 5*inch)
        img_h = (vis_img.drawHeight * 1.05) if vis_img else 25

        add(["FileName", Path(first_row['Path']).name], 25)
        add([vis_img or "Missing", ''], img_h)
        add(['Location Info', f"Floor: {first_row.get('floor', 'N/A')} | View: {first_row.get('view', 'N/A')}"], 25)

        for idx, row in df_record.iterrows():
            crop_img = self._create_rl_image(row['CropPath'], 2*inch, 2*inch) or "Missing"
            add([f"ID: {row.get('ID', f'Defect {idx+1}')}", crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['XYZ/GPS', str(row.get('xyz', 'N/A'))], 20)
            add(['Orientation', str(row.get('orientation', 'N/A'))], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        return data_rows, row_heights


class PDFExporterMeasurement(BasePDFExporter):
    """ 样式 2: 测量报告 """
    def generate_row_content(self, df_record):
        data_rows, row_heights = [], []
        def add(row, h): data_rows.append(row); row_heights.append(h or 20)

        first_row = df_record.iloc[0]
        vis_img = self._create_rl_image(first_row['VisPath'], 4.5*inch)
        img_h = (vis_img.drawHeight + 10) if vis_img else 25

        add(["FileName", Path(first_row['Path']).name], 25)
        add([vis_img or "Missing", ''], img_h)
        add(['Location', f"Floor: {first_row.get('floor','N/A')} | View: {first_row.get('view','N/A')}"], 25)

        for idx, row in df_record.iterrows():
            crop_img = self._create_rl_image(row['CropPath'], 2*inch, 2*inch) or "Missing"
            add([f"ID: {row.get('ID', f'Defect {idx+1}')}", crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            
            w_cm, h_cm, area_cm = row.get('W_cm', 'N/A'), row.get('H_cm', 'N/A'), row.get('Area_cm2', 'N/A')
            add(['Width', f"{w_cm} cm" if w_cm != "N/A" else f"{row.get('W_pix','-')} pix"], 20)
            add(['Height', f"{h_cm} cm" if h_cm != "N/A" else f"{row.get('H_pix','-')} pix"], 20)
            add(['Area', f"{area_cm} cm²" if area_cm != "N/A" else f"{row.get('Area_pix','-')} pix²"], 20)
            add(['Action', row['Action']], 20)
        return data_rows, row_heights


# --- Compact 系列 (流式表格) ---

class PDFExporterCompact(BasePDFExporter):
    """样式 3: 紧凑横向报告"""
    def __init__(self, logo_left, logo_right, target_cls_names=None, max_workers=1):
        super().__init__(logo_left, logo_right, target_cls_names, max_workers)
        # self.pagesize = landscape(A4)
        self.pagesize = A4

    def _get_columns(self):
        """定义列头和列宽"""
        headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image"]
        col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch]
        return headers, col_widths

    def _process_crop_cell(self, path, w_limit, h_limit):
        """辅助：生成单元格图片"""
        img = self._create_rl_image(path, w_limit, h_limit)
        return (img, max(45 if isinstance(self, PDFExporterCompactAuxImage) else 50, img.drawHeight + 6)) if img else ("", 45)

    def _make_row_data(self, idx, row):
        """生成单行数据，子类可覆盖"""
        dim_str = f"H:{float(row.get('H_cm','0')):.1f} * \nW:{float(row.get('W_cm','0')):.1f}"
        img_cell, row_h = self._process_crop_cell(row['CropPath'], 1.3*inch, 1.8*inch)
        
        display_id = str(row.get('ID')) if row.get('ID') is not None else f"DF{idx+1}"
        
        data = [
            str(idx + 1), display_id, f"{row['view']}/\n{row['orientation']}", row['floor'],
            dim_str, row['Level'], row['Category'], row['Action'], img_cell
        ]
        return data, row_h

    def generate_flowables(self, df_record):
        elems = []
        unique_cats = self.target_cls_names if self.target_cls_names else sorted(df_record['Category'].unique()) if 'Category' in df_record.columns else ["Unknown"]
        group_title_style = ParagraphStyle('GroupTitle', parent=self.styles['Heading3'], fontSize=12, spaceBefore=12, spaceAfter=6, textColor=colors.darkblue)
        
        headers, col_widths = self._get_columns()

        print(f"\n[PDF Engine] Compiling {self.__class__.__name__} elements...")
        for cat in tqdm(unique_cats, desc="Processing Categories"):
            sub_df = df_record[df_record['Category'] == cat]
            if sub_df.empty: continue
            if 'ID' in sub_df.columns:
                try: sub_df = sub_df.sort_values(by=['ID'])
                except: pass


            cat_bookmark_title = f"Defect Type: {cat} ({len(sub_df)})"

            key = str(uuid.uuid4())
            if hasattr(self, 'toc'):
                # level=1 代表它是二级标题
                elems.append(TOCEntryNotifier(self.toc, 1, cat_bookmark_title, key))
            elems.append(PDFBookmark(cat_bookmark_title, level=1))
            elems.append(Paragraph(f'<a name="{key}"/>'+cat_bookmark_title, group_title_style))
            # elems.append(Paragraph(cat_bookmark_title, group_title_style))
            
            table_data = [headers]
            row_heights = [30]
            
            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                r_data, r_h = self._make_row_data(local_idx, row)
                table_data.append(r_data)
                row_heights.append(r_h)

            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 9 if len(headers) < 10 else 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ]))
            elems.append(t)
            elems.append(Spacer(1, 15))
        return elems


class PDFExporterCompactAuxImage(PDFExporterCompact):
    """样式 32 (Compact + Aux): 增加 Aux Image 列"""
    def _get_columns(self):
        headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image", "Aux Image"]
        col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch, 1.6*inch]
        return headers, col_widths

    def _make_row_data(self, idx, row):
        dim_str = f"H:{float(row.get('H_cm','0')):.1f} * \nW:{float(row.get('W_cm','0')):.1f}"
        
        img_cell, h1 = self._process_crop_cell(row.get('CropPath'), 1.2*inch, 1.6*inch)
        aux_cell, h2 = self._process_crop_cell(row.get('CropAuxPath'), 1.2*inch, 1.6*inch)
        
        data = [
            str(idx + 1), str(row['ID']), f"{row['view']}/\n{row['orientation']}", row['floor'],
            dim_str, row['Level'], row['Category'], row['Action'], img_cell, aux_cell
        ]
        return data, max(h1, h2)


# --- Context 系列 (按图分组) ---

class PDFExporterWithContext(PDFExporterCompact):
    """样式 4: 图文对照报告"""
    
    def _draw_context_header_images(self, first_row):
        """默认绘制单个大图"""
        vis_img = self._create_rl_image(first_row.get('VisPath'), 9.5*inch, 5.5*inch)
        return [vis_img] if vis_img else [Paragraph("(Visual context image missing)", self.styles["Normal"])]

    def generate_flowables(self, df_record):
        elems = []
        if df_record.empty: return elems

        group_col = 'VisPath' if 'VisPath' in df_record.columns else 'Path'
        unique_images = df_record[group_col].unique()

        headers, col_widths = self._get_columns()
        title_style = ParagraphStyle('ContextTitle', parent=self.styles['Heading2'], backColor=colors.lightgrey, borderPadding=5, spaceAfter=10, textColor=colors.black)
        
        for img_path in unique_images:
            
            sub_df = df_record[df_record[group_col] == img_path]
            if sub_df.empty: continue
            
            first_row = sub_df.iloc[0]

            fname = Path(first_row['Path']).name # 获取文件名

            key = str(uuid.uuid4())
           # 2. 添加目录通知器 (修复报错的关键)
            # 注意：这里我们检查一下 self.toc 是否存在
            if hasattr(self, 'toc'):
                # level=1 代表它是二级标题
                elems.append(TOCEntryNotifier(self.toc, 1, fname, key))
            elems.append(PDFBookmark(fname, level=1))
            elems.append(Paragraph(f'<a name="{key}"/>File: {fname}', title_style))
            # elems.append(Paragraph(f"File: {Path(first_row['Path']).name}", title_style))

            

            # 2. Header Images (Delegate to helper)
            header_imgs = self._draw_context_header_images(first_row)
            for item in header_imgs:
                elems.append(item)
            elems.append(Spacer(1, 15))

            # 3. Table
            table_data = [headers]
            row_heights = [30]
            
            if 'ID' in sub_df.columns:
                try: sub_df = sub_df.sort_values(by=['ID'])
                except: pass

            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                r_data, r_h = self._make_row_data(local_idx, row)
                table_data.append(r_data)
                row_heights.append(r_h)

            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 9 if len(headers) < 10 else 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ]))
            elems.append(t)
            elems.append(PageBreak())
        
        return elems

    def _make_row_data(self, idx, row):
        """Context 模式下的行数据生成"""
        dim_str = f"H:{float(row.get('H_cm','0')):.1f} * \nW:{float(row.get('W_cm','0')):.1f}"
        img_cell, r_h = self._process_crop_cell(row['CropPath'], 1.4*inch, 1.4*inch)
        
        data = [
            str(idx + 1), str(row['ID']), f"{row['view']}/\n{row['orientation']}", row['floor'],
            dim_str, row['Level'], row['Category'], row['Action'], img_cell
        ]
        return data, r_h


class PDFExporterWithContextAuxImage(PDFExporterWithContext):
    """样式 42: 图文对照报告 - 双光版"""
    
    def _draw_context_header_images(self, first_row):
        """重写：绘制并排双图"""
        w_limit, h_limit = 4.6 * inch, 4.0 * inch
        img_vis = self._create_rl_image(first_row.get('VisPath'), w_limit, h_limit) or Paragraph("(Missing)", self.styles["Normal"])
        img_aux = self._create_rl_image(first_row.get('VisAuxPath'), w_limit, h_limit) or Paragraph("(Missing)", self.styles["Normal"])
        
        t = Table([[img_vis, img_aux]], colWidths=[4.75*inch, 4.75*inch])
        t.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
        return [t]

    def _get_columns(self):
        """重写：增加 Aux Image 列"""
        headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image", "Aux Image"]
        col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch, 1.6*inch]
        return headers, col_widths

    def _make_row_data(self, idx, row):
        dim_str = f"H:{float(row.get('H_cm','0')):.1f} * \nW:{float(row.get('W_cm','0')):.1f}"
        
        img_cell, h1 = self._process_crop_cell(row.get('CropPath'), 1.3*inch, 1.3*inch)
        aux_cell, h2 = self._process_crop_cell(row.get('CropAuxPath'), 1.3*inch, 1.3*inch)
        
        data = [
            str(idx + 1), str(row['ID']), f"{row['view']}/\n{row['orientation']}", row['floor'],
            dim_str, row['Level'], row['Category'], row['Action'], img_cell, aux_cell
        ]
        return data, max(h1, h2)