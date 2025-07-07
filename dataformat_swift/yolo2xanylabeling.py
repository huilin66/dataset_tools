import os
import json
from pathlib import Path

import numpy as np
import os.path as osp
from skimage import io
import pandas as pd
import yaml
from tqdm import tqdm


def get_cats(class_file):
    df = pd.read_csv(class_file, header=None, index_col=None, names=['category'])
    cats = df['category'].to_dict()
    return cats

def get_atts(attribute_file):
    with open(attribute_file, 'r') as file:
        attribute_dict = yaml.load(file, Loader=yaml.BaseLoader)['attributes']
    return attribute_dict

def get_image_size(image_file):
    pass
    img = io.imread(image_file)
    height, width = img.shape[:2]
    return width, height

def yolo_to_custom(input_file, output_file, image_file, classes, atts=None):
    custom_data = {
        "version": "2.3.6",
        "flags": {},
        "shapes": [],
    }
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    img_w, img_h = get_image_size(image_file)
    image_size = np.array([img_w, img_h], np.float64)
    for line in lines:
        line = line.strip().split(" ")
        class_index = int(line[0])
        label = classes[class_index]
        if len(line) == 5:
            shape_type = "rectangle"
            cx = float(line[1])
            cy = float(line[2])
            nw = float(line[3])
            nh = float(line[4])
            xmin = int((cx - nw / 2) * img_w)
            ymin = int((cy - nh / 2) * img_h)
            xmax = int((cx + nw / 2) * img_w)
            ymax = int((cy + nh / 2) * img_h)
            points = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
        elif atts is not None and len(line) == 5+len(atts):
            pass
        else:
            shape_type = "polygon"
            if atts is not None:
                points, masks = [], line[2+len(atts):]
                attributes = {}
                for att_id, (k,v) in enumerate(atts.items()):
                    att_value = int(line[att_id + 2])
                    att_value_name = v[att_value]
                    attributes[k] = att_value_name
                for x, y in zip(masks[0::2], masks[1::2]):
                    point = [np.float64(x), np.float64(y)]
                    point = np.array(point, np.float64) * image_size
                    points.append(point.tolist())
            else:
                points, masks = [], line[1:]
                attributes = {}
                for x, y in zip(masks[0::2], masks[1::2]):
                    point = [np.float64(x), np.float64(y)]
                    point = np.array(point, np.float64) * image_size
                    points.append(point.tolist())
        shape = {
            "label": label,
            "shape_type": shape_type,
            "flags": {},
            "points": points,
            "group_id": None,
            "description": None,
            "difficult": False,
            "attributes": attributes,
        }
        custom_data["shapes"].append(shape)
    custom_data["imagePath"] = osp.basename(image_file)
    custom_data["imageData"] = None
    custom_data["imageHeight"] = img_h
    custom_data["imageWidth"] = img_w
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(custom_data, f, indent=2, ensure_ascii=False)


def yolo_to_xanylabeling_dir(yolo_label_dir, images_dir, xanylabeling_label_dir, class_file, attribute_file=None):
    cats = get_cats(class_file)
    atts = get_atts(attribute_file) if attribute_file is not None else None
    os.makedirs(xanylabeling_label_dir, exist_ok=True)

    label_list = os.listdir(yolo_label_dir)
    for label_name in tqdm(label_list):
        json_name = Path(label_name).stem + ".json"
        image_name = Path(label_name).stem + ".jpg"
        label_path = osp.join(yolo_label_dir, label_name)
        image_path = osp.join(images_dir, image_name)
        json_path = osp.join(xanylabeling_label_dir, json_name)
        yolo_to_custom(label_path, json_path, image_path, cats, atts)

if __name__ == "__main__":
    pass
    yolo_label_dir = r"E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_1\label"
    images_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_1\image'
    xanylabeling_labeing_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_1\json'
    class_file = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_1\class.txt'
    # yolo_label_dir = r"E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_2\label"
    # images_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_2\image'
    # xanylabeling_labeing_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_2\json'
    # class_file = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch_2\class.txt'
    attribute_file = r'E:\data\202502_signboard\data_annotation\annotation guide 0510\attribute_v5.yaml'
    yolo_to_xanylabeling_dir(yolo_label_dir, images_dir, xanylabeling_labeing_dir, class_file, attribute_file)