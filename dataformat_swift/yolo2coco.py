import argparse
import json
import os
import sys
import shutil
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm



def addCatItem(coco_data, category_dict):
    for k, v in category_dict.items():
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(k)
        category_item['name'] = v
        coco_data['categories'].append(category_item)


def addImgItem(coco_data, image_set, image_id, file_name, size):
    # image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size[1]
    image_item['height'] = size[0]
    # image_item['license'] = None
    # image_item['flickr_url'] = None
    # image_item['coco_url'] = None
    # image_item['date_captured'] = str(datetime.today())
    coco_data['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(coco_data, annotation_id, object_name, image_id, category_id, bbox, polygon=None):
    annotation_item = dict()
    if polygon is None:
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        annotation_item['segmentation'] = seg
        annotation_item['area'] = bbox[2] * bbox[3]
    else:
        annotation_item['segmentation'] = [polygon.flatten().tolist()]
        area = poly2area(polygon)
        annotation_item['area'] = area
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_item['id'] = annotation_id
    coco_data['annotations'].append(annotation_item)


def xywhn2xywh(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    box = (xmin, ymin, w, h)
    return list(map(int, box))


def poly2area(polygon):
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly2xywh(polygon):
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def yolo2coco(image_path, anno_path, json_path, dst_img_dir=None, seg=False):
    coco_data = dict()
    coco_data['images'] = []
    coco_data['annotations'] = []
    coco_data['categories'] = []

    # category_set = dict()
    image_set = set()

    image_id = 000000
    annotation_id = 0

    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)

    category_set = []
    with open(os.path.join(os.path.dirname(anno_path), 'class.txt'), 'r') as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_id = dict((k, v) for k, v in enumerate(category_set))
    addCatItem(coco_data, category_id)

    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path)]
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    for file in tqdm(files):
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img_path = images[index]
            if dst_img_dir is not None:
                dst_img_path = img_path.replace(image_dir, dst_img_dir)
                shutil.copy(img_path, dst_img_path)
            img = cv2.imread(img_path)
            shape = img.shape
            height, width = img.shape[:2]
            filename = images[index].split(os.sep)[-1]
            image_id += 1
            current_image_id = addImgItem(coco_data, image_set, image_id, filename, shape)
        else:
            continue
        with open(file, 'r') as fid:
            for line in fid.readlines():
                line = [float(x) for x in line.strip().split()]
                category = int(line[0])
                category_name = category_id[category]
                if not seg:
                    bbox = xywhn2xywh((line[1], line[2], line[3], line[4]), shape)
                    addAnnoItem(coco_data, annotation_id, category_name, current_image_id, category, bbox)
                else:
                    polygon = np.array(line[1:]).reshape(-1, 2)
                    polygon[:, 0] = polygon[:, 0] * width
                    polygon[:, 1] = polygon[:, 1] * height
                    box = poly2xywh(polygon)
                    addAnnoItem(coco_data, annotation_id, category_name, current_image_id, category, box, polygon)
                annotation_id += 1

    json.dump(coco_data, open(json_path, 'w'))
    print("class nums:{}".format(len(coco_data['categories'])))
    print("image nums:{}".format(len(coco_data['images'])))
    print("bbox nums:{}".format(len(coco_data['annotations'])))


if __name__ == '__main__':
    pass

    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\images\train'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\train'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_select\annotations\instance_train.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_select\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)
    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\images\val'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\val'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_select\annotations\instance_val.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_select\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)

    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\train'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\train'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_select_add\annotations\instance_train.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_select_add\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)
    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\val'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\val'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_select_add\annotations\instance_val.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_select_add\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)

    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\images\train'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\train'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_add\annotations\instance_train.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_add\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)
    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\images\val'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\val'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco_add\annotations\instance_val.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco_add\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)


    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo\images\train'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo\labels\train'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco\annotations\instance_train.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)
    # image_dir = r'E:\data\1211_monhkok\mk_merge\yolo\images\val'
    # label_dir = r'E:\data\1211_monhkok\mk_merge\yolo\labels\val'
    # json_path = r'E:\data\1211_monhkok\mk_merge\coco\annotations\instance_val.json'
    # dst_img_dir = r'E:\data\1211_monhkok\mk_merge\coco\imgs'
    # parseXmlFilse(image_dir, label_dir, json_path, dst_img_dir)


    # image_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\images\val'
    # label_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\labels\val'
    # json_path = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json'
    # parseXmlFilse(image_dir, label_dir, json_path)
    #
    # image_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\images\train'
    # label_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\labels\train'
    # json_path = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_train.json'
    # parseXmlFilse(image_dir, label_dir, json_path)


    image_dir = r'/data/huilin/data/isds/bd_data/data389_seg/images'
    label_dir = r'/data/huilin/data/isds/bd_data/data389_seg/labels'
    json_path = r'/data/huilin/data/isds/bd_data/data389_seg/instance_all.json'
    yolo2coco(image_dir, label_dir, json_path, seg=True)
