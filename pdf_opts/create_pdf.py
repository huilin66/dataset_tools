import pandas as pd
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, mm



def df2pdf(df, pdf_path, image_path):
    pass

    pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\\Windows\\Fonts\\times.ttf"))
    pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\\Windows\\Fonts\\timesbd.ttf"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="font_title", fontName="TimesNewRoman-Bold", fontSize=22,
                              alignment=reportlab.lib.enums.TA_CENTER, leading=22*1.5))
    styles.add(ParagraphStyle(name="font_text", fontName="TimesNewRoman", fontSize=16, leading=16*1.5))
    styles.add(ParagraphStyle(name="font_section", fontName="TimesNewRoman-Bold", fontSize=20, leading=20*1.5))

    threeline_table = TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 2, colors.black),  # 在标题行下添加边框线
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),  # 在标题行下添加边框线
        ("LINEBELOW", (0, -1), (-1, -1), 2, colors.black),  # 在底部行上添加边框线
        ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
    ])
    new_threeline_table = TableStyle(threeline_table.getCommands())

    blank_table = TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
    ("FONTSIZE", (0, 0), (-1, -1), 16),
    ])

    # 创建一个带有格式的标题
    newline = Spacer(1, 16)
    title = Paragraph("<b>BD-Detection Result Report</b>", styles["font_title"])
    # endregion


    # region part1:input information
    img_result = Image(image_path)
    img_result.drawHeight = 3 * inch * img_result.drawHeight / img_result.drawWidth
    img_result.drawWidth = 3 * inch

    rows_h = [14/40*inch, img_result.drawHeight*1.1]
    data_input = [['property', 'value'], ['input data', img_result]]


    for idx, row in df.iterrows():
        data_input.append([row['property'], row['value']])
        rows_h.append(14/40*inch)

    data_input = Table(data_input, hAlign='CENTER', rowHeights=rows_h)
    new_threeline_table = TableStyle(threeline_table.getCommands())

    data_input.setStyle(new_threeline_table)
    # endregion


    # region part4: create PDF file
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    elements.append(title)
    elements.append(newline)
    elements.append(newline)

    elements.append(data_input)
    elements.append(newline)

    doc.build(elements)

    print('save to', pdf_path)
    # endregion

if __name__ == '__main__':
    pass
    df = pd.read_csv(r'test.csv', header=None, index_col=None, names=['property', 'value'])
    df2pdf(df, 'report.pdf', r'E:\data\2023_defect\road_crack_detection.v2i.yolov9\train\images\2_png_jpg.rf.c09f90face4a925d27bbb84505e95712.jpg')



