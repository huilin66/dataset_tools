import os
import cv2
import ast
import yaml
import json
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import concurrent.futures

RED_BGR = (0, 0, 255)
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
    cats = df['category'].to_list()
    return cats

def get_atts(attribute_file):
    with open(attribute_file, 'r') as file:
        attribute_dict = yaml.load(file, Loader=yaml.BaseLoader)['attributes']
    return attribute_dict

def image_read(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
def image_save(img_path, img):
    cv2.imencode('.png', img)[1].tofile(img_path)

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
    if isinstance(record, str):
        polys = np.array(ast.literal_eval(record), np.float32)
    else:
        polys = np.array(record['masks'], np.float32)
    size = np.array([image_width, image_height] * (len(polys)//2), np.int32)
    polys_sized = polys*size
    polygon_coords = np.array(polys_sized, np.int32).reshape((-1, 2))
    return polygon_coords

def extract_polygon_from_image(image, polygon_coords, crop_method='without_background_keep_shape', color=RED_BGR):
    mask = np.zeros(image.shape[:2], np.uint8)
    top_left_x, top_left_y = np.min(polygon_coords, axis=0)
    bottom_right_x, bottom_right_y = np.max(polygon_coords, axis=0)
    if crop_method.startswith('without_background'):
        cv2.polylines(image, [polygon_coords], isClosed=True, color=color)
        cv2.fillPoly(mask, [polygon_coords], color=255)
        img_copy = cv2.bitwise_and(image, image, mask=mask)
        if crop_method == 'without_background_image_shape':
            img_crop = img_copy
        elif crop_method == 'without_background_box_shape':
            img_crop = img_copy[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    elif crop_method.startswith('with_background'):
        cv2.polylines(image, [polygon_coords], isClosed=True, color=color)
        if crop_method == 'with_background_image_shape':
            img_crop = image
        elif crop_method == 'with_background_box_shape':
            img_crop = image[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)]
    return img_crop, top_left_x, top_left_y

def xywh2poly_crop(record, image, cats, atts=None, annotation=False, crop_method='without_background_keep_shape',
                   filter_no=True, alpha=0.5, tf=1, sf=2/3):
    polygon_coords = polygon_swift(record, image)
    cat_id =  int(record['category'])
    color = COLOR_MAP[cat_id]
    image_crop, top_left_x, top_left_y = extract_polygon_from_image(image, polygon_coords, crop_method=crop_method, color=color)
    if crop_method == 'with_background_image_shape' and annotation:
        text_size = cv2.getTextSize(cats[cat_id], cv2.FONT_HERSHEY_SIMPLEX, sf - 0.1, tf)[0]
        cv2.rectangle(image_crop, (int(top_left_x), int(top_left_y) + 10),
                      (int(top_left_x) + text_size[0] - 15, int(top_left_y) + 7 - text_size[1] + 3),
                      color=COLOR_MAP[cat_id], thickness=-1)
        cv2.putText(image_crop, cats[cat_id], (int(top_left_x), int(top_left_y) + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if atts is not None:
            count = 0
            count2 = 0
            attribute_strs = []

            br_poss = []
            for idx, (k, vs) in enumerate(atts.items()):
                v = vs[int(record[k])]
                text = f'{k}-{v}'
                if filter_no:
                    if not v or v == 'no':
                        continue
                count += 1
                color = (255, 0, 0) if v is not False else (0, 0, 0)
                cv2.putText(image_crop, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                attribute_strs.append(text)
                br_poss.append([int(top_left_x) + text_width - 15, int(top_left_y) + 12 + 10 * count])

            if len(br_poss) > 0:
                br_poss = np.array(br_poss)
                tl_pos = [int(top_left_x), int(top_left_y) + 10]
                br_pos = [np.max(br_poss, axis=0)[0], br_poss[-1][1]]
                box = tl_pos + br_pos
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]) + 15, int(box[3] + 2))
                cv2.rectangle(image_crop, p1, p2, (255, 255, 255))
                overlay = image_crop.copy()
                cv2.rectangle(overlay, p1, p2, (255, 255, 255), -1)
                cv2.addWeighted(overlay, alpha, image_crop, 1 - alpha, 0, image_crop)

                br_poss = []
                for idx, (k, v) in enumerate(atts.items()):
                    v = vs[int(record[k])]
                    text = f'{k}-{v}'
                    if filter_no:
                        if not v or v == 'no':
                            continue
                    count2 += 1
                    color = (255, 0, 0) if v is not False else (0, 0, 0)
                    # text_size = cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12+10*count), cv2.FONT_HERSHEY_SIMPLEX  , 0.5, color, 1)[0]
                    cv2.putText(image_crop, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    attribute_strs.append(text)
                    br_poss.append([int(top_left_x) + text_width - 15, int(top_left_y) + 12 + 10 * count2])
    return image_crop

# endregion


def dict_revert(crop_dict):
    reverted_dict = {value:[] for value in set(crop_dict.values())}
    for key, value in tqdm(crop_dict.items()):
        reverted_dict[value].append(key)
    return reverted_dict

def myolo_crop(image_dir, label_dir, crop_dir, class_file, attribute_file=None, seg=True, annotation=False,
               only_defect=False, save_method='attribute', crop_method='without_background_image_shape'):
    os.makedirs(crop_dir, exist_ok=True)
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
                if save_method == 'all' and only_defect:
                    att_sum = 0
                    for att, levels in atts.items():
                        att_level_int = int(record[att])
                        att_sum += att_level_int
                    if att_sum == 0:
                        continue
                image_crop = xywh2poly_crop(record, image.copy(), crop_method=crop_method, annotation=annotation, cats=cats, atts=atts)
                if image_crop.shape[0]>0 and image_crop.shape[1]>0:
                    save_name = Path(image_name).stem + f'_{idx}' + Path(image_name).suffix
                    if save_method == 'attribute':
                        for att, levels in atts.items():
                            att_level_int = int(record[att])
                            if only_defect and att_level_int == 0:
                                continue
                            att_level = levels[att_level_int]
                            save_path = os.path.join(crop_dir, att, att_level, save_name)
                            image_save(save_path, image_crop)
                    else:
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


def myolo_crop_single(args):
    image_name, image_dir, label_dir, seg, atts, crop_method, annotation, cats, save_method, crop_dir, only_defect = args
    label_name = Path(image_name).stem + '.txt'
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, label_name)
    if not os.path.exists(label_path):
        return {}
    image = image_read(image_path)
    label = label_read(label_path, seg=seg, atts=atts)
    crop_dict = {}

    for idx, record in label.iterrows():
        if seg:
            if save_method == 'all' and only_defect:
                att_sum = 0
                for att, levels in atts.items():
                    att_level_int = int(record[att])
                    att_sum += att_level_int
                if att_sum ==0:
                    continue
            image_crop = xywh2poly_crop(record, image.copy(), crop_method=crop_method, annotation=annotation, cats=cats, atts=atts)
            if image_crop.shape[0]>0 and image_crop.shape[1]>0:
                save_name = Path(image_name).stem + f'_{idx}' + Path(image_name).suffix
                if save_method == 'attribute':
                    for att, levels in atts.items():
                        att_level_int = int(record[att])
                        att_level = levels[att_level_int]
                        if only_defect and att_level_int == 0:
                            continue
                        save_path = os.path.join(crop_dir, att, att_level, save_name)
                        image_save(save_path, image_crop)
                else:
                    save_path = os.path.join(crop_dir,  save_name)
                    image_save(save_path, image_crop)
                crop_dict[save_name] = image_name
        else:
            pass
    return crop_dict

def myolo_crop_mp(image_dir, label_dir, crop_dir, class_file, attribute_file=None, seg=True, annotation=False,
               only_defect=False, save_method='attribute', crop_method='without_background_image_shape'):
    os.makedirs(crop_dir, exist_ok=True)
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

    arg_list = [(image_name, image_dir, label_dir, seg, atts, crop_method, annotation, cats, save_method, crop_dir, only_defect) for image_name in image_list]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(myolo_crop_single, arg_list), total=len(image_list), desc='mask cropping '))

    for result in results:
        crop_dict.update(result)

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
    root_dir = r'E:\data\202502_signboard\data_annotation\dataset\data1422'
    dataset_dir = root_dir
    image_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    image_crop_dir = os.path.join(dataset_dir, 'images_crop')
    class_file = os.path.join(dataset_dir, 'class.txt')
    attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    myolo_crop_mp(image_dir, labels_dir, image_crop_dir, class_file,
               attribute_file=attribute_file, seg=True, annotation=True,
               save_method='all', only_defect=True,
               crop_method='with_background_image_shape')