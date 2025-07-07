# -*- coding: utf-8 -*-
# @File             : report_generation.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2023/12/5 13:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2023/12/5 13:25
'''
@File Description:

'''
import os
import PIL
import cv2
import numpy as np
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import inch, mm
from pathlib import Path
from .visual_utils import draw_box, crop_box, draw_box_ssd, crop_box_ssd
MAX_HEIGHT=100

# region style init
pdfmetrics.registerFont(TTFont("TimesNewRoman", r"C:\\Windows\\Fonts\\times.ttf"))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", r"C:\\Windows\\Fonts\\timesbd.ttf"))
styles1 = getSampleStyleSheet()
styles1.add(ParagraphStyle(name="font_title", fontName="TimesNewRoman-Bold", fontSize=22,
                           alignment=reportlab.lib.enums.TA_CENTER, leading=22 * 1.5))
styles1.add(ParagraphStyle(name="font_text", fontName="TimesNewRoman", fontSize=16, leading=16 * 1.5))
styles1.add(ParagraphStyle(name="font_section", fontName="TimesNewRoman-Bold", fontSize=20, leading=20 * 1.5))

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
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
    ("FONTSIZE", (0, 0), (-1, -1), 16),
])

normal_table = TableStyle([
    ('GRID', (0, 0), (-1, -1), 1, colors.black),  # 设置网格线
    ("FONTNAME", (0, 0), (-1, -1), "TimesNewRoman"),
    ("FONTSIZE", (0, 0), (-1, -1), 14),
    ("VALIGN", (0, 0), (-1, -1), 'MIDDLE'),
])
newline = Spacer(1, 16)
# endregion


def input_prepare(input_info):
    title_input = Paragraph("Input information:", styles1["font_section"])

    if input_info['shape'][0]==input_info['shape'][2] and input_info['shape'][1]==input_info['shape'][3]:
        shape_info = ["%s shape:" % input_info['type'].title(), "%d * %d" % (input_info['shape'][0],
                                                                             input_info['shape'][3],)]
    else:
        shape_info = ["%s shape range:"%input_info['type'].title(), "%d~%d, %d~%d"%(input_info['shape'][0],
                                                                                    input_info['shape'][1],
                                                                                    input_info['shape'][2],
                                                                                    input_info['shape'][3],)]
    data_input = [
        ["Type of Data:", input_info['type'].title()],
        ["Number of %s:"%input_info['type'].title(), input_info['number']],
        shape_info,
    ]

    data_input = Table(data_input, hAlign='LEFT', rowHeights=14/40*inch)
    data_input.setStyle(blank_table)
    return title_input, data_input

def output_prepare(output_info, input_info):
    title_output = Paragraph("Detection information:", styles1["font_section"])

    # summary information
    data_output_summary = [
        ["Model Used for Detection :", output_info['model'].title()],
        ["%s with Defects:"%input_info['type'].title(), output_info['defects']],
        ["%s without Defects:"%input_info['type'].title(), output_info['no defects']],
        ["Detected Defects Summary:", ''],
    ]
    data_output_summary = Table(data_output_summary, hAlign='LEFT', rowHeights=14/40*inch)
    data_output_summary.setStyle(blank_table)

    # defects_sta table
    data_output_defects = [["", "Detected Number"]]
    for k, v in output_info['defects sta'].items():
        data_output_defects.append([k.title(), v])
    data_output_defects = Table(data_output_defects, hAlign='CENTER', rowHeights=14/40*inch)
    data_output_defects.setStyle(threeline_table)
    return title_output, data_output_summary, data_output_defects

def output_prepare_sdd(output_info, input_info):
    title_output = Paragraph("Detection information:", styles1["font_section"])

    # summary information
    data_output_summary = [
        ["Model Used for Detection :", output_info['model'].title()],
        ["%s with Signboard:"%input_info['type'].title(), output_info['with signboard']],
        ["%s without Signboard:"%input_info['type'].title(), output_info['without signboard']],
        ["Detected Signboard Summary:", ''],
    ]
    data_output_summary = Table(data_output_summary, hAlign='LEFT', rowHeights=14/40*inch)
    data_output_summary.setStyle(blank_table)

    # defects_sta table
    data_output_defects = [["", "Detected Number"]]
    for k, v in output_info['detection summary'].items():
        data_output_defects.append([k.title(), v])
    data_output_defects = Table(data_output_defects, hAlign='CENTER', rowHeights=14/40*inch)
    data_output_defects.setStyle(threeline_table)
    return title_output, data_output_summary, data_output_defects

def get_vis_img(img_path, vis_dir, bboxes, label_list, color_list):
    vis_path = os.path.join(vis_dir, Path(img_path).stem+'.png')
    if not os.path.exists(vis_path):
        img = PIL.Image.open(img_path)
        img_draw, _ = draw_box(
            img,
            bboxes.reshape(1, -1),
            labels=label_list,
            colors=color_list,
        )
        img_draw.save(vis_path)
    vis_img = Image(vis_path)
    vis_img.drawHeight = 5 * inch * vis_img.drawHeight / vis_img.drawWidth
    vis_img.drawWidth = 5 * inch
    return vis_img

def get_vis_img_ssd(img_path, vis_dir, bboxes, label_list, color_list, attributes_list):
    vis_path = os.path.join(vis_dir, Path(img_path).stem+'.png')
    if not os.path.exists(vis_path):
        img = cv2.imread(img_path)
        for i in range(len(bboxes)):
            img = draw_box_ssd(
                img,
                bboxes[i],
                labels=label_list,
                colors=color_list,
                attributes=attributes_list[i],
            )
        cv2.imwrite(vis_path, img)
    vis_img = Image(vis_path)
    vis_img.drawHeight = 5 * inch * vis_img.drawHeight / vis_img.drawWidth
    vis_img.drawWidth = 5 * inch
    return vis_img

def get_vis_img_crop(img_path, vis_crop_dir, bboxes, label_list, color_list):
    vis_crop_path = os.path.join(vis_crop_dir, Path(img_path).stem, Path(img_path).stem + '_%d_%d_%d_%d.png' %
                             (bboxes[2], bboxes[3], bboxes[4], bboxes[5]))
    if not os.path.exists(vis_crop_path):
        os.makedirs(os.path.dirname(vis_crop_path), exist_ok=True)
        img = PIL.Image.open(img_path)
        img_crop, _ = crop_box(
            img,
            bboxes.reshape(1, -1),
            labels=label_list,
            colors=color_list,
        )
        img_crop[0].save(vis_crop_path)
    vis_img = Image(vis_crop_path)
    if vis_img.drawHeight <= vis_img.drawWidth:
        vis_img.drawHeight = 2 * inch * vis_img.drawHeight / vis_img.drawWidth
        vis_img.drawWidth = 2 * inch
    else:
        vis_img.drawWidth = 2 * inch * vis_img.drawWidth / vis_img.drawHeight
        vis_img.drawHeight = 2 * inch
    return vis_img

def get_vis_img_crop_ssd(img_path, vis_crop_dir, bbox, label_list, color_list, attributes_list):
    vis_crop_path = os.path.join(vis_crop_dir, Path(img_path).stem, Path(img_path).stem + '_%d_%d_%d_%d.png' %
                                 (bbox[2], bbox[3], bbox[4], bbox[5]))
    if not os.path.exists(vis_crop_path):
        os.makedirs(os.path.dirname(vis_crop_path), exist_ok=True)
        img = cv2.imread(img_path)
        img_crop = crop_box_ssd(
            img,
            bbox,
            labels=label_list,
            colors=color_list,
            attributes=attributes_list,
        )
        cv2.imwrite(vis_crop_path, img_crop)
    vis_img = Image(vis_crop_path)
    if vis_img.drawHeight <= vis_img.drawWidth:
        vis_img.drawHeight = 2 * inch * vis_img.drawHeight / vis_img.drawWidth
        vis_img.drawWidth = 2 * inch
    else:
        vis_img.drawWidth = 2 * inch * vis_img.drawWidth / vis_img.drawHeight
        vis_img.drawHeight = 2 * inch
    return vis_img


def get_bboxes(df, label_list):
    def str_to_list(bbox_str):
        return list(map(float, bbox_str.strip('[]').split()))


    label_to_id = {label.title(): idx for idx, label in enumerate(label_list)}
    cls = df['Category'].map(label_to_id).to_numpy().reshape(-1, 1)
    scores = df['Score'].to_numpy().reshape(-1, 1)
    boxes = np.array(df['Bbox'].apply(str_to_list).tolist())
    bboxes = np.concatenate([cls, scores, boxes], axis=1)
    return bboxes

def get_attributes(df, attribute_list):
    attributes = df[attribute_list].to_dict(orient='records')
    return attributes

def records_prepare(records_info, vis_dir, vis_crop_dir, label_list, color_list):
    title_records = Paragraph("Detailed information:", styles1["font_section"])

    data_records = []
    rows_h = []
    for record_info in records_info:
        if len(record_info)==0:
            continue
        bboxes = get_bboxes(record_info, label_list)
        img_result = get_vis_img(record_info['Path'][0], vis_dir, bboxes, label_list, color_list)

        defect_rec = [
            ["FileName", "%s"%(Path(record_info['Path'][0]).name)],
            [img_result, ''],
            ['Number of Defects', "%d"%len(record_info)],
            ]
        row_h = [
            14/40*inch,
            img_result.drawHeight*1.1,
            14/40*inch*1
        ]
        for idx, record in record_info.iterrows():
            obj_img = get_vis_img_crop(record['Path'], vis_crop_dir, bboxes[idx], label_list, color_list)

            if 'Location' in record.index:
                direction, view, floor = record['Location'].split(',')
                location = 'Palatial Crest'
                address = 'No. 3, Seymour Road, Hong Kong'

                defect_rec += [
                    ['Defect %d'%(idx+1), obj_img],
                    ['Category', record['Category'].title()],
                    ['Level', record['Level'].title()],
                    ['Action', record['Action'].title()],
                    ['Building', location],
                    ['Address', address],
                    ['Direction', direction],
                    ['View', view],
                    ['Floor', floor],
                ]
                row_h += [
                    obj_img.drawHeight * 1.1,
                    14/40*inch*1,
                    14/40*inch*1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                ]
            else:
                defect_rec += [
                    ['Defect %d'%(idx+1), obj_img],
                    ['Category', record['Category'].title()],
                    ['Level', record['Level'].title()],
                    ['Action', record['Action'].title()]
                ]
                row_h += [
                    obj_img.drawHeight * 1.1,
                    14/40*inch*1,
                    14/40*inch*1,
                    14 / 40 * inch * 1
                ]
        data_records += defect_rec
        rows_h += row_h
    data_records = Table(data_records, hAlign='CENTER', rowHeights=rows_h)
    for i in range(len(data_records._cellvalues)):
        if data_records._cellvalues[i][0] == 'FileName':
            new_threeline_table.add('SPAN', (0, i+1), (-1, i+1))  # 合并第一行的所有单元格
    data_records.setStyle(new_threeline_table)
    return title_records, data_records

def get_defect(attributes):
    result = [key for key, value in attributes.items() if value]
    results = ',\n'.join(result)
    return results, len(result)

def records_prepare_ssd(records_info, vis_dir, vis_crop_dir, label_list, color_list, attribute_list):
    title_records = Paragraph("Detailed information:", styles1["font_section"])

    data_records = []
    rows_h = []
    for record_info in records_info:
        if len(record_info)==0:
            continue
        bboxes = get_bboxes(record_info, label_list)
        attributes = get_attributes(record_info, attribute_list)
        img_result = get_vis_img_ssd(record_info['Path'][0], vis_dir, bboxes, label_list, color_list, attributes)

        defect_rec = [
            ["FileName", "%s"%(Path(record_info['Path'][0]).name)],
            [img_result, ''],
            ['Number of Defects', "%d"%len(record_info)],
            ]
        row_h = [
            14/40*inch,
            img_result.drawHeight*1.1,
            14/40*inch*1
        ]
        for idx, record in record_info.iterrows():
            obj_img = get_vis_img_crop_ssd(record['Path'], vis_crop_dir, bboxes[idx], label_list, color_list, attributes[idx])
            defect_str, defect_len = get_defect(attributes[idx])
            if 'Location' in record.index:
                direction, view, floor = record['Location'].split(',')
                location = 'Palatial Crest'
                address = 'No. 3, Seymour Road, Hong Kong'

                defect_rec += [
                    ['Signboard %d'%(idx+1), obj_img],
                    ['Category', record['Category'].title()],
                    ['Defect', defect_str],
                    ['Level', record['Level'].title()],
                    ['Action', record['Action'].title()],
                    ['Building', location],
                    ['Address', address],
                ]
                row_h += [
                    obj_img.drawHeight * 1.1,
                    14/40*inch*1,
                    14/40*inch*1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * defect_len,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                    14 / 40 * inch * 1,
                ]
            else:
                defect_rec += [
                    ['Signboard %d'%(idx+1), obj_img],
                    ['Category', record['Category'].title()],
                    ['Defect', defect_str],
                    ['Level', record['Level'].title()],
                    ['Action', record['Action'].title()]
                ]
                row_h += [
                    obj_img.drawHeight * 1.1,
                    14/40*inch*1,
                    14 / 40 * inch * defect_len,
                    14/40*inch*1,
                    14 / 40 * inch * 1
                ]
        data_records += defect_rec
        rows_h += row_h
    data_records = Table(data_records, hAlign='CENTER', rowHeights=rows_h)
    for i in range(len(data_records._cellvalues)):
        if data_records._cellvalues[i][0] == 'FileName':
            new_threeline_table.add('SPAN', (0, i+1), (-1, i+1))  # 合并第一行的所有单元格
    data_records.setStyle(new_threeline_table)
    return title_records, data_records


def report_get1(report_info, save_path, vis_dir, vis_crop_dir, label_list, color_list, thermal_dir, ref_img_path):
    # region part0:prepare work
    input_info = report_info['input']
    output_info = report_info['output']
    records_info = report_info['records']

    title = Paragraph("<b>AI-Detection Result Report</b>", styles1["font_title"])
    # endregion

    # region part1:input information
    title_input, data_input = input_prepare(input_info)
    # endregion

    # region part2:detection information
    title_output, data_output_summary, data_output_defects = output_prepare(output_info, input_info)
    # endregion

    # region part3:record information
    title_records, data_records = records_prepare(records_info, vis_dir, vis_crop_dir, label_list, color_list)
    # endregion


    # region part4: create PDF file
    doc = SimpleDocTemplate(save_path, pagesize=letter)
    elements = []

    elements.append(title)
    elements.append(newline)
    elements.append(newline)

    elements.append(title_input)
    elements.append(data_input)
    elements.append(newline)

    elements.append(title_output)
    elements.append(data_output_summary)
    elements.append(newline)
    elements.append(data_output_defects)
    elements.append(newline)

    elements.append(PageBreak())
    elements.append(title_records)
    elements.append(newline)
    elements.append(data_records)

    doc.build(elements)

    print('save to', save_path)
    # endregion

def report_get2(report_info, save_path, vis_dir, vis_crop_dir, label_list, color_list, thermal_dir, ref_img_path):
    title = Paragraph("<b>BD-Detection Result Report</b>", styles1["font_title"])
    doc = SimpleDocTemplate(save_path, pagesize=letter)
    elements = []
    elements.append(title)
    elements.append(PageBreak())

    records_info = report_info['records']
    ref_img = Image(ref_img_path)
    assert ref_img.drawHeight > ref_img.drawWidth, 'shape error!'
    ref_img.drawHeight = 6 * inch
    ref_img.drawWidth = 1 * inch

    col_widths = [3.5 * inch, 3.5 * inch]

    for record_info in records_info:
        for idx, record in record_info.iterrows():
            rgb_vis_path = os.path.join(vis_dir, Path(record['Path']).stem+'.png')
            t_path = os.path.join(thermal_dir, Path(record['Path']).name)
            rgb_vis_img = Image(rgb_vis_path)
            t_img = Image(t_path)

            rgb_vis_img.drawWidth = 2.5 * inch * rgb_vis_img.drawWidth / rgb_vis_img.drawHeight
            rgb_vis_img.drawHeight = 2.5 * inch
            t_img.drawWidth = 2.5 * inch * t_img.drawWidth / t_img.drawHeight
            t_img.drawHeight = 2.5 * inch

            p1_data = [
                ['Defect ID: DF%04d' % (idx + 1), 'Severity:%s'%record['Level']],
                ['Location: %s'%record['Location'],  'Defect Type: %s'%record['DefectType']],
                ['Follow-up action: %s'%record['Action'].title(), ''],
                ['Remark:%s'%record['Category'], '']
            ]
            h1 = [
                14 / 40 * inch,
                1.5 * 14 / 40 * inch,
                14 / 40 * inch,
                14 / 40 * inch,
            ]
            p1_table = Table(p1_data, hAlign='CENTER', rowHeights=h1, colWidths=col_widths)
            p1_table.setStyle(normal_table)


            p2_data = [
                ['Image1', record['View']],
                [rgb_vis_img,ref_img],
                ['Image2', ''],
                [t_img,'']
            ]
            h2 = [
                14 / 40 * inch,
                rgb_vis_img.drawHeight * 1.2,
                14 / 40 * inch,
                t_img.drawHeight * 1.2,
            ]

            p2_table = Table(p2_data, hAlign='CENTER', rowHeights=h2, colWidths=col_widths)
            new_table_style = TableStyle(normal_table.getCommands())  # 复制 normal_table 的样式
            new_table_style.add('SPAN', (1, 1), (1, 3))  # 添加合并单元格的样式
            p2_table.setStyle(new_table_style)

            elements.append(p1_table)
            elements.append(newline)
            elements.append(p2_table)
            elements.append(PageBreak())
    doc.build(elements)

def report_get_sdd(report_info, save_path, vis_dir, vis_crop_dir, label_list, color_list, thermal_dir, ref_img_path, attribute_list):
    # region part0:prepare work
    input_info = report_info['input']
    output_info = report_info['output']
    records_info = report_info['records']

    title = Paragraph("<b>Signboard Defect Detection Result Report</b>", styles1["font_title"])
    # endregion

    # region part1:input information
    title_input, data_input = input_prepare(input_info)
    # endregion

    # region part2:detection information
    title_output, data_output_summary, data_output_defects = output_prepare_sdd(output_info, input_info)
    # endregion

    # region part3:record information
    title_records, data_records = records_prepare_ssd(records_info, vis_dir, vis_crop_dir, label_list, color_list, attribute_list)
    # endregion


    # region part4: create PDF file
    doc = SimpleDocTemplate(save_path, pagesize=letter)
    elements = []

    elements.append(title)
    elements.append(newline)
    elements.append(newline)

    elements.append(title_input)
    elements.append(data_input)
    elements.append(newline)

    elements.append(title_output)
    elements.append(data_output_summary)
    elements.append(newline)
    elements.append(data_output_defects)
    elements.append(newline)

    elements.append(PageBreak())
    elements.append(title_records)
    elements.append(newline)
    elements.append(data_records)

    doc.build(elements)

    print('save to', save_path)
    # endregion




if __name__ == '__main__':
    pass
