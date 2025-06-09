import os
import cv2
import yaml
import json
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

COLOR_MAP = [
    (255, 42, 4),
    # (235, 219, 11),
    # (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (255, 255, 0),
    (0, 192, 38),
    (179, 255, 1),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (11, 255, 162),
    (0, 0, 255),
]

# region basic tools

def get_cats(class_file):
    df = pd.read_csv(class_file, header=None, index_col=None, names=['category'])
    cats = df['category'].to_dict()
    return cats

def get_atts(attribute_file):
    with open(attribute_file, 'r') as file:
        attribute_dict = yaml.load(file, Loader=yaml.BaseLoader)['attributes']
    return attribute_dict

def image_read(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img

def image_save(img_path, img):
    cv2.imwrite(img_path, img)

def label_read(label_path, seg=False, atts=None):
    def poly2xywh(mask):
        mask = np.array([mask[::2], mask[1::2]])
        x_min, y_min = np.min(mask, axis=1)
        x_max, y_max = np.max(mask, axis=1)
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        width = x_max - x_min
        height = y_max - y_min
        return [x_center, y_center, width, height]

    names = ['category']
    if atts is not None:
        names += ['attribute_len'] + list(atts.keys())
    names += ['center_x', 'center_y', 'width', 'height']
    if seg:
        names += ['masks']

    if seg:
        df = pd.DataFrame(None, columns=names)
        with open(label_path, 'r') as f:
            data = f.readlines()
            for id_line, line in enumerate(data):
                parts = line.strip().split(' ')
                category = int(parts[0])

                if atts is not None:
                    att_len = int(parts[1])
                    atts = list(map(float, parts[2:2 + att_len]))
                    polygons = list(map(float, parts[2 + att_len:]))
                    xywh = poly2xywh(polygons)
                    record = [category, att_len] + atts + xywh + [polygons]
                else:
                    polygons = list(map(float, parts[1:]))
                    xywh = poly2xywh(polygons)
                    record = [category] + xywh + [polygons]
                df.loc[len(df)] = record
    else:
        df = pd.read_csv(label_path, header=None, index_col=None, sep=' ', names=names)
    return df

# endregion


# region extract tools

def calculate_polygon_area(points):
    """使用鞋带公式计算多边形面积"""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0

def polygon_swift(record, image):
    image_height, image_width = image.shape[:2]
    polys = np.array(record['masks'], np.float32)
    size = np.array([image_width, image_height] * (len(polys)//2), np.int32)
    polys_sized = polys*size
    polygon_coords = np.array(polys_sized, np.int32).reshape((-1, 2))
    return polygon_coords

def extract_polygon_from_image(image, polygon_coords, crop_method='without_background_keep_shape'):
    mask = np.zeros(image.shape[:2], np.uint8)

    if crop_method.startswith('without_background'):
        cv2.polylines(image, [polygon_coords], isClosed=False, color=255)
        cv2.fillPoly(mask, [polygon_coords], color=255)
        img_copy = cv2.bitwise_and(image, image, mask=mask)
        if crop_method == 'without_background_image_shape':
            img_crop = img_copy
        elif crop_method == 'without_background_box_shape':
            top_left_x, top_left_y = np.min(polygon_coords, axis=0)
            bottom_right_x, bottom_right_y = np.max(polygon_coords, axis=0)
            img_crop = img_copy[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    elif crop_method.startswith('with_background'):
        pass
    return img_crop

def xywh2poly_crop(record, image, crop_method='without_background_keep_shape'):
    polygon_coords = polygon_swift(record, image)
    image_crop = extract_polygon_from_image(image, polygon_coords, crop_method=crop_method)

    return image_crop

# endregion


def dict_revert(crop_dict):
    reverted_dict = {value:[] for value in set(crop_dict.values())}
    for key, value in tqdm(crop_dict.items()):
        reverted_dict[value].append(key)
    return reverted_dict

def myolo_crop(image_dir, label_dir, crop_dir, class_file, attribute_file=None, seg=True, save_method='attribute', crop_method='without_background_image_shape'):
    os.makedirs(crop_dir)
    cats = get_cats(class_file)
    atts = get_atts(attribute_file) if attribute_file is not None else None

    if save_method == 'attribute' and atts is not None:
        for att in atts:
            att_dir = os.path.join(crop_dir, att)
            for idx, level in enumerate(atts[att]):
                att_level_dir = os.path.join(att_dir, level)
                os.makedirs(att_level_dir, exist_ok=True)
    elif save_method == 'category':
        for cat in cats:
            cat_dir = os.path.join(crop_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)
    elif save_method == 'attribute_category' and atts is not None:
        pass

    crop_dict = {}
    image_list = os.listdir(image_dir)
    for img_idx, image_name in enumerate(tqdm(image_list, desc='mask cropping ')):
        label_name = Path(image_name).stem + '.txt'
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_name)
        if not os.path.exists(label_path):
            continue
        image = image_read(image_path)
        label = label_read(label_path, seg=seg, atts=atts)
        for idx, record in label.iterrows():
            if seg:
                image_crop = xywh2poly_crop(record, image.copy(), crop_method=crop_method)
                if image_crop.shape[0]>0 and image_crop.shape[1]>0:
                    if save_method == 'attribute':
                        for att, levels in atts.items():
                            att_level = levels[int(record[att])]
                            save_path = os.path.join(crop_dir, att, att_level, Path(image_name).stem + f'_{idx}' + Path(image_name).suffix)
                            image_save(save_path, image_crop)
                    else:
                        save_name = Path(image_name).stem + f'_{idx}' + Path(image_name).suffix
                        save_path = os.path.join(crop_dir,  save_name)
                        image_save(save_path, image_crop)
                        crop_dict[save_name] = image_name
            else:
                pass
    crop_result_path = crop_dir+'.json'
    with open(crop_result_path, 'w') as f:
        json.dump(crop_dict, f, ensure_ascii=False, indent=4)

    with open(crop_result_path, 'r') as f:
        crop_dict = json.load(f)
    crop_result_revert_path = crop_dir+'_revert.json'
    crop_dict_revert = dict_revert(crop_dict)
    with open(crop_result_revert_path, 'w') as f:
        json.dump(crop_dict_revert, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    pass
    # root_dir = r'/data/huilin/data/isds/ps_data/0527'
    # image_dir = os.path.join(root_dir, 'images')
    # label_dir = os.path.join(root_dir, 'images_seg_infer', 'labels')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    #
    # crop_dir = os.path.join(root_dir, 'images_crop_box')
    # myolo_crop(image_dir, label_dir, crop_dir, class_file,
    #            attribute_file=None, seg=True,
    #            save_method='all',
    #            crop_method='without_background_box_shape')

    root_dir = r'/data/huilin/data/isds/ps_data/0606'
    image_folder = os.path.join(root_dir, 'merge_dir')
    yolo_infer_folder = os.path.join(root_dir, 'merge_dir_seg_infer', 'labels')
    crop_folder = os.path.join(root_dir, 'merge_dir_crop')
    caption_folder = os.path.join(root_dir, 'caption')
    class_file = os.path.join(root_dir, 'class.txt')
    llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    myolo_crop(image_folder, yolo_infer_folder, crop_folder, class_file,
               attribute_file=None, seg=True,
               save_method='all',
               crop_method='without_background_box_shape')