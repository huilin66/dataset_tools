import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
from skimage import io
import numpy as np
import pandas as pd
from tqdm import tqdm
class_mapping = {
    'Boeing737': 0,
    'Boeing747': 1,
    'Boeing777': 2,
    'Boeing787': 3,
    'C919': 4,
    'A220': 5,
    'A321': 6,
    'A330': 7,
    'A350': 8,
    'ARJ21': 9,
    'other-airplane': 10,
    'A320/321': 11,
    'Boeing737-800': 12,
    'other': 13,
}
mapping_class = {
    0: 'Boeing737',
    1: 'Boeing747',
    2: 'Boeing777',
    3: 'Boeing787',
    4: 'C919',
    5: 'A220',
    6: 'A321',
    7: 'A330',
    8: 'A350',
    9: 'ARJ21',
    10: 'other-airplane',
    11: 'A320/321',
    12: 'Boeing737-800',
    13: 'other',
}
class_mapping_rgb = {
    'Boeing737': 0,
    'Boeing747': 1,
    'Boeing777': 2,
    'Boeing787': 3,
    'C919': 4,
    'A220': 5,
    'A321': 6,
    'A330': 7,
    'A350': 8,
    'ARJ21': 9,
    'other-airplane': 10,
}
class_mapping_sar = {
    'A220': 0,
    'A330': 1,
    'A320/321': 2,
    'Boeing737-800': 3,
    'Boeing787': 4,
    'ARJ21': 5,
    'other': 6,
}

def convert_tz_to_yolo(gt_dir, label_dir, img_dir, image_dir, class_mapping):
    def coord_str2num(coord_str):
        strs = coord_str.split(',')
        num1 = int(strs[0].split('.')[0])
        num2 = int(strs[1].split('.')[0])
        return num1, num2

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    gt_list = os.listdir(gt_dir)
    for gt_file in tqdm(gt_list):
        if gt_file.endswith('.xml'):
            gt_path = os.path.join(gt_dir, gt_file)
            img_path = os.path.join(img_dir, gt_file.replace('xml', 'tif'))
            image_path = os.path.join(image_dir, gt_file.replace('xml', 'png'))
            label_path = os.path.join(label_dir, gt_file.replace('xml', 'txt'))
            try:
                img = io.imread(img_path)
                width, height = img.shape[1], img.shape[0]
            except Exception as e:
                print(e, img_path)
                continue

            # io.imsave(image_path, img)

            tree = ET.parse(gt_path)
            root = tree.getroot()

            df = pd.DataFrame(None, columns=['cat_id', 'x', 'y', 'h', 'w'])

            # size = root.findall('size')[0]
            # width = int(size.find('width').text)
            # height = int(size.find('height').text)

            objects = root.findall('objects')
            if len(objects) > 0:
                object_list = objects[0].findall('object')
                for object in object_list:
                    class_name = object.find('possibleresult').find('name').text
                    class_id = class_mapping[class_name]

                    points = object.findall('points')[0]
                    points_list = points.findall('point')
                    point_nums = []
                    for point in points_list:
                        point_num = coord_str2num(point.text)
                        point_nums.append(point_num)
                    point_nums = np.array(point_nums)
                    x_min = np.min(point_nums[:, 0])
                    x_max = np.max(point_nums[:, 0])
                    y_min = np.min(point_nums[:, 1])
                    y_max = np.max(point_nums[:, 1])
                    x_center = (x_max + x_min) / 2
                    y_center = (y_max + y_min) / 2
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    x_center /= width
                    y_center /= height
                    box_w /= width
                    box_h /= height
                    df.loc[len(df)] = [class_id, x_center, y_center, box_w, box_h]

            df.to_csv(label_path, header=False, index=False, sep=' ')

def convert_yolo_to_tz(input_dir, output_dir, image_dir, mapping_class):
    def prettify_xml(elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    os.makedirs(output_dir, exist_ok=True)

    input_list = os.listdir(input_dir)
    for input_file in tqdm(input_list):
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file.replace('txt', 'xml'))
        image_path = os.path.join(image_dir, input_file.replace('txt', 'tif'))
        img = io.imread(image_path)
        width, height, depth = img.shape[1], img.shape[0], img.shape[2]

        df = pd.read_csv(input_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'h', 'w'], sep=' ')

        annotation = ET.Element("annotation")
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "filename").text = str(input_file)
        ET.SubElement(source, "origin").text = 'Optical' if depth==3 else 'SAR'
        research = ET.SubElement(annotation, "research")
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        objects = ET.SubElement(annotation, "objects")
        for idx, row in df.iterrows():
            cat_name = mapping_class[int(row['cat_id'])]
            xmin = int((row['x'] - row['w'] / 2) * width)
            ymin = int((row['y'] - row['h'] / 2) * height)
            xmax = int((row['x'] + row['w'] / 2) * width)
            ymax = int((row['y'] + row['h'] / 2) * height)

            object = ET.SubElement(objects, "object")
            ET.SubElement(object, "coordinate").text = 'pixel'
            ET.SubElement(object, "type").text = 'rectangle'
            ET.SubElement(object, "description").text = 'None'
            possibleresult = ET.SubElement(object, "possibleresult")
            ET.SubElement(possibleresult, "name").text = cat_name
            points = ET.SubElement(object, "points")
            ET.SubElement(points, "point").text = '%.6f,%.6f'%(xmin, ymin)
            ET.SubElement(points, "point").text = '%.6f,%.6f'%(xmax, ymin)
            ET.SubElement(points, "point").text = '%.6f,%.6f'%(xmax, ymax)
            ET.SubElement(points, "point").text = '%.6f,%.6f'%(xmin, ymax)
            ET.SubElement(points, "point").text = '%.6f,%.6f'%(xmin, ymin)

        # tree = ET.ElementTree(annotation)
        # tree.write(output_path)
        xml_str = prettify_xml(annotation)
        with open(output_path, "w") as f:
            f.write(xml_str)

if __name__ == '__main__':
    pass
    # gt_dir = r'E:\data\tp\multi_modal_airplane_train\gt'  # VOC格式标注文件夹
    # yolo_label_dir = r'E:\data\tp\multi_modal_airplane_train\labels'  # 输出YOLO格式标注文件夹
    # img_dir = r'E:\data\tp\multi_modal_airplane_train\img'
    # image_dir = r'E:\data\tp\multi_modal_airplane_train\image'
    # convert_tz_to_yolo(gt_dir, yolo_label_dir, img_dir, image_dir, class_mapping)

    result_dir = r'E:\data\tp\multi_modal_airplane_train\infer_result\labels'
    output_dir = r'E:\data\tp\multi_modal_airplane_train\infer_result_final'
    img_dir = r'E:\data\tp\multi_modal_airplane_train\infer_result'
    convert_yolo_to_tz(result_dir, output_dir, img_dir, mapping_class)