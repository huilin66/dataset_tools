# exporters/base_exporter.py
from ast import Break
from collections import defaultdict
import concurrent.futures
import os
from pathlib import Path
import time
from copy import deepcopy

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table, TableStyle
from tqdm import tqdm

import config

class BasePDFExporter:
    def __init__(self, logo_left, logo_right, target_cls_names=None, max_workers=1):
        self._init_fonts()
        self._init_styles()
        self.logo_left = logo_left
        self.logo_right = logo_right
        self.target_cls_names = target_cls_names
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def generate_row_content(self, df_record):
        raise NotImplementedError("Subclasses must implement generate_row_content")

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

        self.table_style_common = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), 14),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        
        # 定义三种表格样式供子类调用
        self.style_blank = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ("FONTNAME", (0, 0), (-1, -1), self.FONT_REGULAR)])
        self.style_threeline = TableStyle([("LINEABOVE", (0, 0), (-1, 0), 2, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black), ("VALIGN", (0, 0), (-1, -1), 'MIDDLE')])

    def _draw_header_logos(self, canvas, doc):
        """
        回调函数：在每一页渲染时被调用，用于绘制页眉 Logo。
        """
        canvas.saveState() # 保存当前画布状态，避免影响正文
        
        # 获取当前页面尺寸（兼容横向和纵向）
        page_w, page_h = doc.pagesize
        
        # === 配置区域 ===
        # 假设 Logo 路径存在类属性里，或者这里硬编码，或者从全局 config 读取
        # 你也可以在 export 时把路径存为 self.logo_left_path
        left_logo_path = getattr(self, 'logo_left', None)   # 左上角 Logo
        right_logo_path = getattr(self, 'logo_right', None) # 右上角 Logo
        
        target_h = 0.45 * inch  # 设定 Logo 高度
        margin = 20            # 页边距
        # ================

        from reportlab.lib.utils import ImageReader

        # 1. 绘制左上角 Logo
        if left_logo_path and os.path.exists(left_logo_path):
            try:
                img = ImageReader(left_logo_path)
                iw, ih = img.getSize()
                aspect = iw / ih if ih > 0 else 1
                
                draw_h = target_h
                draw_w = draw_h * aspect
                
                # 坐标：左边距, 页面高度 - 顶部边距 - 图片高度
                x = margin
                y = page_h - margin - draw_h
                
                canvas.drawImage(left_logo_path, x, y, width=draw_w, height=draw_h, mask='auto')
            except Exception as e:
                print(f"Error drawing left logo: {e}")

        # 2. 绘制右上角 Logo
        if right_logo_path and os.path.exists(right_logo_path):
            try:
                img = ImageReader(right_logo_path)
                iw, ih = img.getSize()
                aspect = iw / ih if ih > 0 else 1
                
                draw_h = target_h
                draw_w = draw_h * aspect
                
                # 坐标：页面宽度 - 右边距 - 图片宽度, 页面高度 - 顶部边距 - 图片高度
                x = page_w - margin - draw_w
                y = page_h - margin - draw_h
                
                canvas.drawImage(right_logo_path, x, y, width=draw_w, height=draw_h, mask='auto')
            except Exception as e:
                print(f"Error drawing right logo: {e}")

        canvas.restoreState() # 恢复状态

    def generate_row_content(self, df_record):
        """
        【抽象方法】这是子类必须实现的核心差异方法。
        输入：单张图片的 DataFrame
        输出：(data_rows, row_heights)
        """
        raise NotImplementedError("Subclasses must implement generate_row_content")

    def export(self, report_data, save_path):
        """
        异步导出方法。
        调用此方法后，会立即返回一个 Future 对象，不会阻塞主线程。
        实际的 PDF 生成和写入将在后台线程中执行。
        """
        print(f"[{time.strftime('%H:%M:%S')}] PDF Task queued: {os.path.basename(save_path)}")
        
        # 将实际的导出任务提交给线程池
        # submit(函数名, 参数1, 参数2...)
        future = self.executor.submit(self._execute_export_sync, report_data, save_path)
        
        # 你可以添加回调函数来处理完成后的通知（可选）
        future.add_done_callback(lambda f: self._on_export_complete(f, save_path))
        
        return future

    def _on_export_complete(self, future, save_path):
        try:
            future.result() # 如果任务中有异常，会在这一步抛出
            print(f"Finished: report exported to {save_path}\n") # 可以在这里做日志记录
        except Exception as e:
            print(f"!!! Error exporting {save_path}: {e}")


    def _execute_export_sync(self, report_data, save_path):
        # =========================================================
        # 这里的内容就是你原来 export() 函数的所有代码
        # =========================================================
        print(f"[{time.strftime('%H:%M:%S')}] PDF Generation started in background ({self.__class__.__name__})...\n")

        target_pagesize = getattr(self, 'pagesize', letter)
        doc = SimpleDocTemplate(save_path, pagesize=target_pagesize)
        
        elements = []

        # --- 1. 通用表头与摘要 ---
        self._add_summary_pages(elements, report_data)

        # --- 2. 详细内容 ---
        elements.append(Paragraph("Detailed Information:", self.styles["font_section"]))
        elements.append(Spacer(1, 10))

        records_df_list = report_data['records']
        
        # 模式 A: 流式布局
        if hasattr(self, 'generate_flowables'):
            # 注意：在线程中打印 tqdm 可能会导致进度条错乱，建议去掉 tqdm 或使用特定参数
            # 这里简化为直接遍历
            for df_record in records_df_list:
                if df_record.empty: continue
                record_elements = self.generate_flowables(df_record)
                elements.extend(record_elements)
                elements.append(PageBreak())

        # 模式 B: 统一大表格布局
        else:
            all_data_rows = []
            all_row_heights = []

            for df_record in records_df_list:
                if df_record.empty: continue
                rows, heights = self.generate_row_content(df_record)
                all_data_rows.extend(rows)
                all_row_heights.extend(heights)

            if all_data_rows:
                min_len = min(len(all_data_rows), len(all_row_heights))
                all_data_rows = all_data_rows[:min_len]
                all_row_heights = all_row_heights[:min_len]

                from reportlab.lib.units import inch
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

        try:
            # 这一步是最耗时的，现在它在后台线程运行
            print(f"[{time.strftime('%H:%M:%S')}] Compiling PDF {os.path.basename(save_path)}...")
            doc.build(
                elements,
                onFirstPage=self._draw_header_logos, 
                onLaterPages=self._draw_header_logos
                )
            print(f"[{time.strftime('%H:%M:%S')}] Success! Saved to {save_path}")
            
        except Exception as e:
            print(f"Error in thread: {e}")
            import traceback
            traceback.print_exc()
            raise e # 抛出异常以便 Future 捕获

    def _get_view_from_record(self, df):
        """辅助函数：尝试从 DataFrame 中获取 View 信息，如果缺失则从路径推断"""
        if df.empty: return "Unknown"
        # 1. 尝试直接读取列
        if 'view' in df.columns:
            val = df.iloc[0]['view']
            if val and str(val) != 'nan': return str(val)
        
        # 2. 回退：从路径推断 (假设路径结构为 .../V01/IMG.jpg)
        try:
            path_obj = Path(df.iloc[0]['Path'])
            parent_name = path_obj.parent.name
            # 简单的启发式：如果是 V 开头且后面是数字
            if parent_name.startswith('V') and parent_name[1:].isdigit():
                return parent_name
        except:
            pass
        return "Unknown"


    def _get_summary_table_style(self):
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ])

    def _add_summary_pages(self, elements, report_data):
        """
        汇总页生成 (v5版 - 修复 View/Floor 统计单一的问题):
        逻辑变更：不再假设一个 DF 只有一个 View，而是逐行读取 View/Floor 进行统计。
        """
        import re 

        input_info = report_data['input']
        output_info = report_data['output']
        records_list = report_data['records']

        # --- 1. 标题与基础信息 ---
        elements.append(Paragraph("<b>Project Summary Report</b>", self.styles["font_title"]))

        elements.append(Paragraph(f"Basic Information:", self.styles["font_section"]))

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


        # 2.1 Defect/levels Definitions
        explanation_json = input_info.get('explanation_json', {})

        categories = deepcopy(explanation_json["Category"])
        categories_des = categories.pop("Description")

        elements.append(Paragraph("Defect Definitions:", self.styles["font_section"]))
        elements.append(Paragraph(categories_des, self.styles["font_text"]))
        elements.append(Spacer(1, 10))

        # 1. 准备表头
        table_rows = [["Defect Type", "Description"]]
        
        # 2. 准备样式 (用于长文本自动换行)
        # 使用较小的字号(10)以容纳更多文字
        desc_style = ParagraphStyle(
            'GlossaryDesc', 
            parent=self.styles['Normal'], 
            fontSize=10, 
            leading=12
        )

        # 3. 遍历字典填充数据
        for defect_type, description in categories.items():
            # 使用 Paragraph 包装描述文本，确保自动换行
            p_desc = Paragraph(str(description), desc_style)
            table_rows.append([defect_type, p_desc])

        # 4. 设置列宽 (根据页面方向调整)
        # 如果是横向(Compact)，第二列给宽一点；如果是纵向，给窄一点
        is_landscape = isinstance(self, PDFExporterCompact)
        col_widths = [2.0 * inch, 7.0 * inch] if is_landscape else [1.5 * inch, 4.5 * inch]

        # 5. 创建表格
        t_gloss = Table(table_rows, colWidths=col_widths)
        
        # 6. 表格样式
        t_gloss.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), # 表头灰底
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),     # 网格线
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),     # 表头加粗
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),               # 内容顶对齐 (重要！否则短的那列会居中)
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(t_gloss)
        elements.append(Spacer(1, 20)) # 表格后留白



        levels = deepcopy(explanation_json["Levels"])
        levels_des = levels.pop("Description")

        elements.append(Paragraph("Level Definitions:", self.styles["font_section"]))
        elements.append(Paragraph(levels_des, self.styles["font_text"]))
        elements.append(Spacer(1, 10))

        # 1. 准备表头
        table_rows = [["Levels", "Description"]]
        
        # 2. 准备样式 (用于长文本自动换行)
        # 使用较小的字号(10)以容纳更多文字
        desc_style = ParagraphStyle(
            'GlossaryDesc', 
            parent=self.styles['Normal'], 
            fontSize=10, 
            leading=12
        )

        # 3. 遍历字典填充数据
        for level_type, description in levels.items():
            # 使用 Paragraph 包装描述文本，确保自动换行
            p_desc = Paragraph(str(description), desc_style)
            table_rows.append([level_type, p_desc])

        # 4. 设置列宽 (根据页面方向调整)
        # 如果是横向(Compact)，第二列给宽一点；如果是纵向，给窄一点
        is_landscape = isinstance(self, PDFExporterCompact)
        col_widths = [2.0 * inch, 7.0 * inch] if is_landscape else [1.5 * inch, 4.5 * inch]

        # 5. 创建表格
        t_gloss = Table(table_rows, colWidths=col_widths)
        
        # 6. 表格样式
        t_gloss.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), # 表头灰底
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),     # 网格线
            ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),     # 表头加粗
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),               # 内容顶对齐 (重要！否则短的那列会居中)
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(t_gloss)
        elements.append(Spacer(1, 20)) # 表格后留白

        elements.append(PageBreak())


        # 2. views map png
        elements.append(Paragraph("Views Direction Map", self.styles["font_title"]))
        views_map_path = input_info['views_png_path']
        img = RLImage(views_map_path)
        # 设置最大限制 (根据页面方向调整宽度)
        # 如果是横向(Compact/Aux)，宽限大一点；如果是纵向(Basic)，宽限小一点
        is_landscape = isinstance(self, PDFExporterCompact)
        max_w = 8.5 * inch if is_landscape else 6.0 * inch
        max_h = 5.0 * inch
        
        # 保持比例缩放
        img_w, img_h = img.imageWidth, img.imageHeight
        aspect = img_h / img_w if img_w > 0 else 1.0
        
        draw_w = max_w
        draw_h = draw_w * aspect
        
        if draw_h > max_h:
            draw_h = max_h
            draw_w = draw_h / aspect
        
        img.drawWidth = draw_w
        img.drawHeight = draw_h
        
        elements.append(img)

        elements.append((PageBreak()))

        # --- 2. 数据聚合 (Data Aggregation) ---
        
        # 定义 4 个统计容器 (Key -> Severity -> Count)
        stats_cat_lev = defaultdict(lambda: defaultdict(int))   # 类别
        stats_view_lev = defaultdict(lambda: defaultdict(int))  # 视角 (View)
        stats_ori_lev = defaultdict(lambda: defaultdict(int))   # 方向 (Direction)
        stats_floor_lev = defaultdict(lambda: defaultdict(int)) # 楼层 (Floor)
        
        defined_cats = self.target_cls_names if self.target_cls_names else report_data.get('defined_categories', [])
        defined_floors = report_data.get('defined_floors', [])[::-1]
        
        all_categories = set(defined_cats)
        all_floors = set(defined_floors)
        all_views = set()
        all_orientations = set()

        # 遍历每一个 DataFrame (哪怕只有一个巨大的 DF 也没关系)
        for df in records_list:
            if df.empty: continue
            
            # 🔥 核心修复：移除了 df.iloc[0] 读取 View/Floor 的逻辑
            # 改为在 iterrows 内部逐行读取
            
            for _, row in df.iterrows():
                # 1. 提取基础信息
                cat = row['Category']
                level = row['Level']
                
                # 2. 🔥 逐行提取 View (确保 v30, v32... 都能被读到)
                v = str(row.get('view', 'Unknown')).strip()
                if v == 'nan' or not v: v = "Unknown"
                
                # 3. 🔥 逐行提取 Floor
                fl = str(row.get('floor', 'Unknown')).strip()
                if fl == 'nan' or not fl: fl = "Unknown"

                # 4. 🔥 逐行提取 Orientation
                o = str(row.get('orientation', 'Unknown')).strip()
                if o == 'nan' or not o: o = "Unknown"

                # 收集所有出现的 Key 以便后续排序
                all_categories.add(cat)
                all_views.add(v)
                all_floors.add(fl)
                if o != "Unknown":
                    all_orientations.add(o)

                # 5. 填充统计数据 (使用当前行的 v, fl, o)
                stats_cat_lev[level][cat] += 1
                stats_view_lev[level][v] += 1     # 现在这里是当前行的 View，不再是第一行的 View
                stats_floor_lev[level][fl] += 1   # 同理
                
                if o != "Unknown":
                    stats_ori_lev[level][o] += 1

        # --- 3. 排序逻辑 (Sorting) ---

        # 自然排序辅助函数
        def natural_keys(text):
            return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(text))]

        # A. 类别排序
        if defined_cats:
            remaining = sorted(list(all_categories - set(defined_cats)))
            sorted_cats = defined_cats + remaining
        else:
            sorted_cats = sorted(list(all_categories))
            
        # B. View 排序
        sorted_views = sorted(list(all_views), key=natural_keys)
        
        # C. Direction 排序
        dir_order = {'N':1, 'NE':2, 'E':3, 'SE':4, 'S':5, 'SW':6, 'W':7, 'NW':8}
        sorted_oris = sorted(list(all_orientations), key=lambda x: dir_order.get(x, 99))

        # D. 楼层排序 (智能排序)
        floor_order = {floor: index for index, floor in enumerate(defined_floors)}
            
        sorted_floors = sorted(list(all_floors), key=lambda floor: floor_order.get(floor, float('inf')))

        # --- 4. 表格生成函数 ---
        def add_level_table(title, row_keys, stats_dict, col1_name):
            if not row_keys: return
            
            elements.append(Paragraph(title, self.styles["font_section"]))
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
            t.setStyle(self._get_summary_table_style())
            elements.append(t)
            elements.append(Spacer(1, 25))

        # === 5. 输出所有表格 ===
        add_level_table("1. Summary by Defect Type:", sorted_cats, stats_cat_lev, "Defect Type")
        add_level_table("2. Summary by View:", sorted_views, stats_view_lev, "View ID")
        if sorted_oris:
            add_level_table("3. Summary by Direction:", sorted_oris, stats_ori_lev, "Direction")
        add_level_table("4. Summary by Floor:", sorted_floors, stats_floor_lev, "Floor")

        elements.append(PageBreak())


class PDFExporterBasic(BasePDFExporter):
    """ 样式 0: 基础报告 """
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
        add(['Defect Count', str(len(df_record))], 25)

        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            defect_id = str(row.get('ID', f'Defect {idx+1}'))
            add([f'ID: {defect_id}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        return data_rows, row_heights


class PDFExporterDetailed(BasePDFExporter):
    """ 样式 1: 详细报告 """
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
        loc_info = f"Floor: {first_row.get('floor', 'N/A')} | View: {first_row.get('view', 'N/A')}"
        add(['Location Info', loc_info], 25)

        for idx, row in df_record.iterrows():
            crop_path = row['CropPath']
            crop_img = RLImage(crop_path, width=2*inch, height=2*inch) if os.path.exists(crop_path) else "Missing"
            defect_id = str(row.get('ID', f'Defect {idx+1}'))
            add([f'ID: {defect_id}', crop_img], 2*inch + 10)
            add(['Category', row['Category']], 20)
            xyz = row.get('xyz', 'N/A')
            ori = row.get('orientation', 'N/A')
            add(['XYZ/GPS', str(xyz)], 20)
            add(['Orientation', str(ori)], 20)
            add(['Level', row['Level']], 20)
            add(['Action', row['Action']], 20)
            add(['Score', f"{row['Score']:.2f}"], 20)
        return data_rows, row_heights


class PDFExporterMeasurement(BasePDFExporter):
    """ 样式 2: 测量报告 """
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
    1. 自动从文件路径提取 View (修复原始代码丢失元数据的问题)。
    2. 显示 GPS/Alt 和 Direction。
    """
    def __init__(self, logo_left, logo_right, target_cls_names=None, max_workers=1):
        super().__init__(logo_left, logo_right, target_cls_names=target_cls_names, max_workers=max_workers)
        self.pagesize = landscape(letter)

    # 找到 PDFExporterCompact 类，替换其中的 generate_flowables 方法
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

        # 2. 全景大图 (保持原有逻辑，带宽高限制)
        vis_path = first_row['VisPath']
        if os.path.exists(vis_path):
            vis_img = RLImage(vis_path)
            limit_w, limit_h = 8.5 * inch, 5.0 * inch
            img_w, img_h = vis_img.imageWidth, vis_img.imageHeight
            aspect = img_h / img_w if img_w > 0 else 1.0
            draw_w = limit_w
            draw_h = draw_w * aspect
            if draw_h > limit_h:
                draw_h = limit_h
                draw_w = draw_h / aspect
            vis_img.drawWidth = draw_w
            vis_img.drawHeight = draw_h
            elems.append(vis_img)
            elems.append(Spacer(1, 10))

        # ==========================================
        # 3. 分组生成表格 (Group by Defect Type)
        # ==========================================
        
        # 获取所有出现的类别并排序
        if 'Category' in df_record.columns:
            unique_cats = sorted(df_record['Category'].unique())
        else:
            unique_cats = ["Unknown"]

        # 定义样式
        cell_style = ParagraphStyle('CellStyle', parent=self.styles['Normal'], fontSize=8, leading=10, alignment=1)
        # 分组标题样式 (深蓝色，稍微大一点)
        group_title_style = ParagraphStyle(
            'GroupTitle', 
            parent=self.styles['Heading3'], 
            fontSize=12, 
            spaceBefore=12, 
            spaceAfter=6, 
            textColor=colors.darkblue
        )
        
        # 定义列头和列宽 (所有分组共用)
        headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image"]
        # 定义列宽 (总宽约 10 inch)
        col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch]

        # --- 循环遍历每个类别 ---
        print("\n[PDF Engine] Compiling Compact Report elements...")
        # tqdm 会自动计算速度和预计剩余时间 (ETA)
        for cat in tqdm(unique_cats, desc="Processing Categories"):
            # 1. 筛选当前类别的数据
            sub_df = df_record[df_record['Category'] == cat]
            if sub_df.empty: continue
            
            # 2. 排序 (组内按 ID 排序)
            if 'ID' in sub_df.columns:
                try:
                    sub_df = sub_df.sort_values(by=['ID'])
                except: 
                    pass # 如果 ID 是字符串混杂，可能会排序失败，保持原样
            
            # 3. 添加 分组标题 (例如: "Defect Type: Spalling (Count: 5)")
            elems.append(Paragraph(f"Defect Type: {cat} (Count: {len(sub_df)})", group_title_style))
            
            # 4. 构建当前类别的表格数据
            table_data = [headers]
            row_heights = [30]
            
            # 遍历该组的每一行
            # local_idx 用于显示组内的序号 1, 2, 3...
            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                
                # --- 尺寸处理 ---
                w_val, h_val = row.get('W_cm', 'N/A'), row.get('H_cm', 'N/A')
                dim_str = f"H:{float(h_val):.1f} * \nW:{float(w_val):.1f}"

                # --- 图片处理 ---
                crop_path = row['CropPath']
                img_cell = ""
                row_h = 40
                if os.path.exists(crop_path):
                    img = RLImage(crop_path)
                    target_w, max_h = 1.3 * inch, 1.8 * inch
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

                # --- 等级处理 ---
                lvl = row['Level']
                display_lvl = 'Minor' if lvl == 'Slight' else ('Major' if lvl == 'Serious' else lvl)
                
                # --- ID 显示 ---
                # 如果有真实ID就显示真实ID，否则显示 DF+序号
                real_id = row.get('ID')
                display_id = f"{real_id}" if real_id is not None else f"DF{local_idx+1}"

                table_row = [
                    str(local_idx + 1),
                    row['ID'],
                    f"{row['view']}/\n{row['orientation']}",
                    row['floor'],
                    dim_str,
                    row['Level'],
                    row['Category'],
                    row['Action'],
                    img_cell
                ]
                table_data.append(table_row)
                row_heights.append(row_h)

            # 5. 生成表格并添加到元素列表
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
            # 每个表格后加一点空白
            elems.append(Spacer(1, 15))

        return elems 


class PDFExporterCompactAuxImage(PDFExporterCompact):
    """
    样式 32 (Compact + Aux): 
    基于 PDFExporterCompact (样式3)，在表格右侧增加 'Aux Image' 列。
    适用于双光吊舱数据，以紧凑列表形式展示 RGB 和 Thermal 截图。
    """
    def generate_flowables(self, df_record):
        elems = []
        
        if 'Category' in df_record.columns:
            unique_cats = sorted(df_record['Category'].unique())
        else:
            unique_cats = ["Unknown"]

        # 定义样式
        cell_style = ParagraphStyle('CellStyle', parent=self.styles['Normal'], fontSize=8, leading=10, alignment=1)
        group_title_style = ParagraphStyle(
            'GroupTitle', 
            parent=self.styles['Heading3'], 
            fontSize=12, 
            spaceBefore=12, 
            spaceAfter=6, 
            textColor=colors.darkblue
        )
        
        # --- 【修改点 1】: 增加 Aux Image 列头，并调整列宽以适应页面 ---
        headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image", "Aux Image"]
        # 定义列宽 (总宽约 10 inch)
        col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch, 1.6*inch]

        from tqdm import tqdm
        print("\n[PDF Engine] Compiling Compact (Aux) Report elements...")
        
        for cat in tqdm(unique_cats, desc="Processing Categories"):
            sub_df = df_record[df_record['Category'] == cat]
            if sub_df.empty: continue
            
            if 'ID' in sub_df.columns:
                try: sub_df = sub_df.sort_values(by=['ID'])
                except: pass
            
            elems.append(Paragraph(f"Defect Type: {cat} (Count: {len(sub_df)})", group_title_style))
            
            table_data = [headers]
            row_heights = [30]
            
            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                
                # --- 尺寸处理 ---
                w_val, h_val = row.get('W_cm', 'N/A'), row.get('H_cm', 'N/A')
                dim_str = f"H:{float(h_val):.1f} * \nW:{float(w_val):.1f}"

                # --- 辅助函数：处理截图 (复用逻辑) ---
                def process_crop(path_key):
                    c_path = row.get(path_key, '')
                    if c_path and os.path.exists(c_path):
                        img = RLImage(c_path)
                        target_w, max_h = 1.2 * inch, 1.6 * inch # 稍微改小一点适应列宽
                        aspect = img.imageHeight / img.imageWidth if img.imageWidth > 0 else 1.0
                        calc_h = target_w * aspect
                        if calc_h > max_h:
                            img.drawHeight = max_h
                            img.drawWidth = max_h / aspect
                        else:
                            img.drawWidth = target_w
                            img.drawHeight = calc_h
                        return img, max(45, img.drawHeight + 6)
                    return "", 45

                # --- 图片处理 (Visible & Aux) ---
                img_cell, h1 = process_crop('CropPath')
                aux_img_cell, h2 = process_crop('CropAuxPath')
                
                # 行高取两者最大值
                row_h = max(h1, h2)
                table_row = [
                    str(local_idx + 1),
                    row['ID'],
                    f"{row['view']}/\n{row['orientation']}",
                    row['floor'],
                    dim_str,
                    row['Level'],
                    row['Category'],
                    row['Action'],
                    img_cell,      # 原 Vis Crop
                    aux_img_cell   # 新增 Aux Crop
                ]
                table_data.append(table_row)
                row_heights.append(row_h)

            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
            
            style_list = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 8), # 字体调小适应内容
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]
            t.setStyle(TableStyle(style_list))
            
            elems.append(t)
            elems.append(Spacer(1, 15))

        return elems


class PDFExporterWithContext(PDFExporterCompact):
    """
    样式 4: 图文对照报告 (Contextual Report)
    逻辑: 
    1. 按源文件(图片)分组。
    2. 顶部显示该图片带框的检测结果大图(VisPath)。
    3. 下方紧接该图片内所有缺陷的详细列表表格。
    """
    def generate_flowables(self, df_record):
        elems = []
        if df_record.empty: return elems

        # 1. 按照可视化的全景图路径进行分组
        # 这样确保同一张大图的缺陷会聚在一起
        # 注意：如果 VisPath 缺失，回退到 Path
        group_col = 'VisPath' if 'VisPath' in df_record.columns else 'Path'
        
        # 获取所有唯一的图片路径（保持原始顺序）
        unique_images = df_record[group_col].unique()

        for img_path in unique_images:
            # 筛选出当前这张图的所有缺陷数据
            sub_df = df_record[df_record[group_col] == img_path]
            if sub_df.empty: continue
            
            # --- A. 标题部分 ---
            first_row = sub_df.iloc[0]
            fname = Path(first_row['Path']).name
            
            # 标题样式
            title_style = ParagraphStyle(
                'ContextTitle', 
                parent=self.styles['Heading2'], 
                backColor=colors.lightgrey, 
                borderPadding=5,
                spaceAfter=10,
                textColor=colors.black
            )
            elems.append(Paragraph(f"File: {fname}", title_style))

            # --- B. 插入全景大图 (Context Image) ---
            vis_path = first_row['VisPath']
            if os.path.exists(vis_path):
                vis_img = RLImage(vis_path)
                
                # 设置大图的最大尺寸 (Landscape 页面宽度较大)
                # 留出页边距，假设可用宽度约 9.5 inch
                max_w, max_h = 9.5 * inch, 5.5 * inch
                
                img_w, img_h = vis_img.imageWidth, vis_img.imageHeight
                aspect = img_h / img_w if img_w > 0 else 1.0
                
                draw_w = max_w
                draw_h = draw_w * aspect
                
                # 如果高度超标，则按高度限制缩放
                if draw_h > max_h:
                    draw_h = max_h
                    draw_w = draw_h / aspect
                
                vis_img.drawWidth = draw_w
                vis_img.drawHeight = draw_h
                
                # 图片居中 (使用 Table 包裹或者直接 append)
                elems.append(vis_img)
                elems.append(Spacer(1, 15)) # 图片和表格之间的间距
            else:
                elems.append(Paragraph("(Visual context image missing)", self.styles["Normal"]))
                elems.append(Spacer(1, 15))

            # --- C. 构建缺陷表格 ---
            # 定义表头
            headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image"]
            # 定义列宽 (总宽约 10 inch)
            col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch]
            
            table_data = [headers]
            row_heights = [30] # 表头高度

            # 组内按 ID 排序
            if 'ID' in sub_df.columns:
                try: sub_df = sub_df.sort_values(by=['ID'])
                except: pass

            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                # 1. 尺寸文本
                w_val, h_val = row.get('W_cm', 'N/A'), row.get('H_cm', 'N/A')
                dim_str = f"H:{float(h_val):.1f} * \nW:{float(w_val):.1f}"

                # 2. 局部截图 (Crop Image)
                crop_path = row['CropPath']
                img_cell = ""
                this_row_h = 45 # 默认行高
                
                if os.path.exists(crop_path):
                    c_img = RLImage(crop_path)
                    c_max_w, c_max_h = 1.4 * inch, 1.4 * inch
                    c_aspect = c_img.imageHeight / c_img.imageWidth if c_img.imageWidth > 0 else 1.0
                    
                    c_draw_h = c_max_w * c_aspect
                    if c_draw_h > c_max_h:
                        c_img.drawHeight = c_max_h
                        c_img.drawWidth = c_max_h / c_aspect
                    else:
                        c_img.drawWidth = c_max_w
                        c_img.drawHeight = c_draw_h
                    
                    img_cell = c_img
                    this_row_h = max(45, c_img.drawHeight + 6)

                row_data = [
                    str(local_idx + 1),
                    row['ID'],
                    f"{row['view']}/\n{row['orientation']}",
                    row['floor'],
                    dim_str,
                    row['Level'],
                    row['Category'],
                    row['Action'],
                    img_cell
                ]

                table_data.append(row_data)
                row_heights.append(this_row_h)

            # --- D. 生成表格样式并添加到 elems ---
            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
            
            style_list = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), # 表头背景
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),     # 网格线
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),     # 表头字体
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),             # 居中
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),            # 垂直居中
                ('FONTSIZE', (0, 0), (-1, -1), 9),                 # 字体大小
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]
            t.setStyle(TableStyle(style_list))
            elems.append(t)
            
            # --- E. 每张大图结束后分页 ---
            elems.append(PageBreak())

        return elems


class PDFExporterWithContextAuxImage(PDFExporterWithContext):
    """
    样式 42: 图文对照报告 (Contextual Report) - 双光版
    逻辑: 
    1. 按源文件(图片)分组。
    2. 顶部并排显示: 左侧可见光大图(VisPath), 右侧辅助/红外大图(VisAuxPath)。
    3. 下方表格增加 Crop Aux Image 列。
    """
    def generate_flowables(self, df_record):
        elems = []
        if df_record.empty: return elems

        # 1. 按照可视化的全景图路径进行分组
        group_col = 'VisPath' if 'VisPath' in df_record.columns else 'Path'
        
        # 获取所有唯一的图片路径（保持原始顺序）
        unique_images = df_record[group_col].unique()

        for img_path in unique_images:
            # 筛选出当前这张图的所有缺陷数据
            sub_df = df_record[df_record[group_col] == img_path]
            if sub_df.empty: continue
            
            # --- A. 标题部分 ---
            first_row = sub_df.iloc[0]
            fname = Path(first_row['Path']).name
            
            # 标题样式
            title_style = ParagraphStyle(
                'ContextTitle', 
                parent=self.styles['Heading2'], 
                backColor=colors.lightgrey, 
                borderPadding=5,
                spaceAfter=10,
                textColor=colors.black
            )
            elems.append(Paragraph(f"File: {fname}", title_style))

            # --- B. 插入全景大图 (Context Images: Left Vis, Right Aux) ---
            vis_path = first_row.get('VisPath', '')
            vis_aux_path = first_row.get('VisAuxPath', '')
            
            # 定义单个大图的最大尺寸 (页面宽度一分为二，减去一点间隙)
            # 假设总可用宽 9.5 inch -> 每张图最大宽约 4.6 inch
            max_top_w, max_top_h = 4.6 * inch, 4.0 * inch

            # 内部函数：处理图片缩放
            def process_top_image(path):
                if path and os.path.exists(path):
                    img = RLImage(path)
                    img_w, img_h = img.imageWidth, img.imageHeight
                    aspect = img_h / img_w if img_w > 0 else 1.0
                    
                    draw_w = max_top_w
                    draw_h = draw_w * aspect
                    
                    if draw_h > max_top_h:
                        draw_h = max_top_h
                        draw_w = draw_h / aspect
                    
                    img.drawWidth = draw_w
                    img.drawHeight = draw_h
                    return img
                else:
                    return Paragraph("(Missing)", self.styles["Normal"])

            # 处理左图 (Vis) 和 右图 (Aux)
            img_vis_obj = process_top_image(vis_path)
            img_aux_obj = process_top_image(vis_aux_path)

            # 使用 Table 将两张图并排布局
            top_table_data = [[img_vis_obj, img_aux_obj]]
            top_table = Table(top_table_data, colWidths=[4.75*inch, 4.75*inch])
            top_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            
            elems.append(top_table)
            elems.append(Spacer(1, 15)) # 图片和表格之间的间距

            # --- C. 构建缺陷表格 ---
            # 定义表头 (新增 Crop Aux Image)
            headers = ["No.", "ID", "Direction", "Floor", "Size(cm²)", "Level", "Defect", "Action", "Defect Image", "Aux Image"]
            # 定义列宽 (总宽约 10 inch)
            col_widths = [0.4*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 1.6*inch, 0.8*inch, 1.6*inch, 1.6*inch]
            
            
            table_data = [headers]
            row_heights = [30] # 表头高度

            # 组内按 ID 排序
            if 'ID' in sub_df.columns:
                try: sub_df = sub_df.sort_values(by=['ID'])
                except: pass

            for local_idx, (_, row) in enumerate(sub_df.iterrows()):
                # 1. 尺寸文本
                w_val, h_val = row.get('W_cm', 'N/A'), row.get('H_cm', 'N/A')
                dim_str = f"H:{float(h_val):.1f} * \nW:{float(w_val):.1f}"
            
                # 内部函数：处理小截图缩放
                def process_crop_image(c_path):
                    if c_path and os.path.exists(c_path):
                        c_img = RLImage(c_path)
                        c_max_w, c_max_h = 1.3 * inch, 1.3 * inch # 稍微限制尺寸以适应单元格
                        c_aspect = c_img.imageHeight / c_img.imageWidth if c_img.imageWidth > 0 else 1.0
                        
                        c_draw_h = c_max_w * c_aspect
                        if c_draw_h > c_max_h:
                            c_img.drawHeight = c_max_h
                            c_img.drawWidth = c_max_h / c_aspect
                        else:
                            c_img.drawWidth = c_max_w
                            c_img.drawHeight = c_draw_h
                        return c_img, max(45, c_img.drawHeight + 6)
                    return "", 45

                # 2. 局部截图 (Crop Image & Crop Aux Image)
                crop_path = row.get('CropPath', '')
                crop_aux_path = row.get('CropAuxPath', '')

                img_cell, h1 = process_crop_image(crop_path)
                aux_img_cell, h2 = process_crop_image(crop_aux_path)
                
                # 当前行高取最大值
                this_row_h = max(h1, h2)

                row_data = [
                    str(local_idx + 1),
                    row['ID'],
                    f"{row['view']}/\n{row['orientation']}",
                    row['floor'],
                    dim_str,
                    row['Level'],
                    row['Category'],
                    row['Action'],
                    img_cell,      # 原 Vis Crop
                    aux_img_cell   # 新增 Aux Crop
                ]
                
                table_data.append(row_data)
                row_heights.append(this_row_h)

            # --- D. 生成表格样式并添加到 elems ---
            t = Table(table_data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
            
            style_list = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), # 表头背景
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),     # 网格线
                ('FONTNAME', (0, 0), (-1, 0), self.FONT_BOLD),     # 表头字体
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),             # 居中
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),            # 垂直居中
                ('FONTSIZE', (0, 0), (-1, -1), 8),                 # 字体调小一点以防换行过多
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]
            t.setStyle(TableStyle(style_list))
            elems.append(t)
            
            # --- E. 每张大图结束后分页 ---
            elems.append(PageBreak())

        return elems
