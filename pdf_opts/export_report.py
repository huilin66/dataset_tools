import os
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, mm


def get_img(img_path, enlarge_size=5*inch):
    vis_img = Image(img_path)
    vis_img.drawHeight = enlarge_size * vis_img.drawHeight / vis_img.drawWidth
    vis_img.drawWidth = enlarge_size
    vis_img.hAlign = 'CENTER'
    return vis_img


def get_table(data, table_style):
    table = Table(data, colWidths=[2*inch, 2*inch, 2*inch])
    table.setStyle(table_style)
    return table

def get_records_data(records, title_style, table_style, empty_line, page_symbol):
    results = []
    for record in records:
        title = Paragraph(f"record #{record[0]}", title_style)
        img = get_img(record[1])

        table_content = [
            ['category', record[2]],
            ['img_name', os.path.basename(record[1])],
            ['height', record[3]],
        ]
        table = Table(table_content, colWidths=[1.5 * inch, 3.5 * inch])
        table.setStyle(table_style)

        results.append(title)
        results.append(empty_line)
        results.append(img)
        results.append(empty_line)
        results.append(table)
        results.append(page_symbol)
    return results

def get_report(pdf_path, img_path_pcd, list_summary, records):

    # region part0: definition
    empty_line = Spacer(1, 0.25*inch)
    page_symbol = PageBreak()
    threeline_table_style = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 2, colors.black),  # 在标题行下添加边框线
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),  # 在标题行下添加边框线
        ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),  # 在底部行上添加边框线
        ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
    ])
    normal_table_style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.aliceblue),
    ])
    pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\\Windows\\Fonts\\times.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\\Windows\\Fonts\\timesbd.ttf"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="font_title", fontName="TimesNewRoman-Bold", fontSize=22,
                              alignment=reportlab.lib.enums.TA_LEFT, leading=22 * 1.5))
    # styles.add(ParagraphStyle(name="font_text", fontName="TimesNewRoman", fontSize=16, leading=16 * 1.5))
    # styles.add(ParagraphStyle(name="font_section", fontName="TimesNewRoman-Bold", fontSize=20, leading=20 * 1.5))

    # endregion


    # region part1: page 1 data
    img = get_img(img_path_pcd)

    table = get_table(list_summary, threeline_table_style)
    # endregion

    # region part2: record page data
    results = get_records_data(records, styles["font_title"], normal_table_style, empty_line, page_symbol)


    # region part3: create PDF file
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    elements.append(img)
    elements.append(empty_line)
    elements.append(table)
    elements.append(page_symbol)

    elements += results
    doc.build(elements)

    print('save to', pdf_path)
    # endregion

if __name__ == '__main__':
    pass
    data = [
        ['category', 'number', 'condition'],
        ['class1', 10, True],
        ['class2', 10, True],
        ['class3', 10, True],
        ['class4', 10, False],
    ]
    records = [
        [0, 'img.png', 'class2', 1.01],
        [1, 'img.png', 'class3', 3.01],
        [3, 'img.png', 'class12', 2.01],
        [4, 'img.png', 'class6', 3.01],
    ]
    get_report('report.pdf', r'img.png', data, records)


    '/nfsv4/23039356r/repository/ultralytics/runs/mdetect/exp_yolo10x_head1231_20/weights'
