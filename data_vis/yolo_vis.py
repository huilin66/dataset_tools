import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import pandas as pd

colormap = [
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
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
    (11, 255, 162)
]

# region class tools
def get_cats(class_file):
    df = pd.read_csv(class_file, header=None, index_col=None, names=['category'])
    cats = df['category'].to_dict()
    return cats
# endregion

# region attribute tools
def attribute2label(label, attribute_values, attributes, attribute_len):
    attribute_labels = np.zeros(attribute_len)
    idx = 0
    if len(attribute_values)>0:
        for k,v in attributes[label].items():
            assert len(v)>1
            for i in range(1, len(v)):
                if attribute_values[k]==v[i]:
                    attribute_labels[idx] = 1
                idx += 1
    return attribute_labels

def get_attribute_len(attributes):
    attribute_len = 0
    for k, v in attributes.items():
        attribute_len += len(v)-1
    return attribute_len

def get_attribute(attribute_dict, gt_attribute):
    attributes = {}
    idx = 0
    attribute_len = get_attribute_len(attribute_dict)
    assert attribute_len == len(gt_attribute)
    if isinstance(gt_attribute[0], str):
        gt_attribute = [int(gt_value) for gt_value in gt_attribute]
    for k, v in attribute_dict.items():
        assert len(v) > 1
        attributes[k] = False
        for i in range(1, len(v)):
            if gt_attribute[idx] == 1:
                attributes[k] = v[i]
            idx += 1
    return attributes
# endregion


def xywh2xyxy(x, w1, h1, img, cats, crop=True, attributes=None, filter_no=False, alpha=0.5, tf=1, sf=2/3):
    label, x, y, w, h = x

    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    if crop:
        img_crop = img[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)].copy()
    else:
        img_crop = None
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[int(label)], 2)

    text_size = cv2.getTextSize(cats[int(label)], cv2.FONT_HERSHEY_SIMPLEX, sf - 0.1, tf)[0]

    # Draw the background rectangle
    cv2.rectangle(img,
                  (int(top_left_x), int(top_left_y)+10),
                  (int(top_left_x)+text_size[0]-15, int(top_left_y)+7-text_size[1]+3),
                  color=colormap[int(label)], thickness=-1)
    cv2.putText(img, cats[int(label)], (int(top_left_x), int(top_left_y)+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if attributes is not None:
        count = 0
        count2 = 0
        attribute_strs = []

        br_poss = []
        for idx, (k, v) in enumerate(attributes.items()):
            text = f'{k}-{v}'
            if filter_no:
                if not v:
                    continue
            count += 1
            color = (255, 0, 0) if v is not False else (0, 0, 0)
            cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            attribute_strs.append(text)
            br_poss.append([int(top_left_x)+text_width-15, int(top_left_y) + 12+10*count])

        if len(br_poss)>0:
            br_poss = np.array(br_poss)
            tl_pos = [int(top_left_x), int(top_left_y)+10]
            br_pos = [np.max(br_poss, axis=0)[0], br_poss[-1][1]]
            box = tl_pos + br_pos
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2])+15, int(box[3]+2))
            cv2.rectangle(img, p1, p2, (255, 255, 255))
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            br_poss = []
            for idx, (k, v) in enumerate(attributes.items()):
                text = f'{k}-{v}'
                if filter_no:
                    if not v:
                        continue
                count2 += 1
                color = (255, 0, 0) if v is not False else (0, 0, 0)
                # text_size = cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12+10*count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)[0]
                cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                attribute_strs.append(text)
                br_poss.append([int(top_left_x) + text_width - 15, int(top_left_y) + 12 + 10 * count2])
    else:
        attribute_strs = None
    return img, img_crop, attribute_strs

def xywh2poly(x, w, h, img, cats, crop=True, attributes=None, filter_no=False, alpha=0.5, tf=1, sf=2/3):
    label, polypos = int(float(x[0])),x[1:]
    polys = []
    for i in range(0, len(polypos), 2):
        pos1 = float(polypos[i]) * w
        pos2 = float(polypos[i+1]) * h
        polys.append([pos1, pos2])

    polys = np.array(polys, np.int32)
    if crop:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.polylines(mask, [polys], isClosed=True, color=255)
        cv2.fillPoly(mask, [polys], color=255)
        img_crop = cv2.bitwise_and(img, img, mask=mask)
        top_left_x, top_left_y = np.min(polys, axis=0)
        bottom_right_x, bottom_right_y = np.max(polys, axis=0)
        img_crop = img_crop[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)].copy()
    else:
        img_crop = None


    cv2.polylines(img, [polys], isClosed=True, color=colormap[int(label)], thickness=2)
    mask = img.copy()
    cv2.fillPoly(mask, [polys], color=colormap[int(label)])
    cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

    text_size = cv2.getTextSize(cats[int(label)], cv2.FONT_HERSHEY_SIMPLEX, sf - 0.1, tf)[0]
    cv2.rectangle(img,
                  (int(top_left_x), int(top_left_y)+10),
                  (int(top_left_x)+text_size[0]-15, int(top_left_y)+7-text_size[1]+3),
                  color=colormap[int(label)], thickness=-1)
    cv2.putText(img, cats[int(label)], (int(top_left_x), int(top_left_y)+7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    print(cats[int(label)], int(top_left_y),int(bottom_right_y), int(top_left_x),int(bottom_right_x), int(bottom_right_y)-int(top_left_y), int(bottom_right_x)-int(top_left_x))
    if attributes is not None:
        count = 0
        count2 = 0
        attribute_strs = []

        br_poss = []
        for idx, (k, v) in enumerate(attributes.items()):
            text = f'{k}-{v}'
            if filter_no:
                if not v:
                    continue
            count += 1
            color = (255, 0, 0) if v is not False else (0, 0, 0)
            cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            attribute_strs.append(text)
            br_poss.append([int(top_left_x)+text_width-15, int(top_left_y) + 12+10*count])

        if len(br_poss)>0:
            br_poss = np.array(br_poss)
            tl_pos = [int(top_left_x), int(top_left_y)+10]
            br_pos = [np.max(br_poss, axis=0)[0], br_poss[-1][1]]
            box = tl_pos + br_pos
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2])+15, int(box[3]+2))
            cv2.rectangle(img, p1, p2, (255, 255, 255))
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            br_poss = []
            for idx, (k, v) in enumerate(attributes.items()):
                text = f'{k}-{v}'
                if filter_no:
                    if not v:
                        continue
                count2 += 1
                color = (255, 0, 0) if v is not False else (0, 0, 0)
                # text_size = cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12+10*count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)[0]
                cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                attribute_strs.append(text)
                br_poss.append([int(top_left_x) + text_width - 15, int(top_left_y) + 12 + 10 * count2])
    else:
        attribute_strs = None

    return img, img_crop, attribute_strs

def yolo_data_vis(img_folder, label_folder, output_folder, class_file, crop_dir=None, seg=False):
    cats = get_cats(class_file)
    img_list = os.listdir(img_folder)
    img_list.sort()
    label_list = os.listdir(label_folder)
    label_list.sort()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if crop_dir is not None and not os.path.exists(crop_dir):
        os.mkdir(crop_dir)
        for cat in cats.values():
            os.mkdir(os.path.join(crop_dir, cat))

    for i in tqdm(range(len(img_list))):
        image_path = os.path.join(img_folder, img_list[i])
        label_path = os.path.join(label_folder, label_list[i])
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            if seg:
                lb = [x.split() for x in f.read().strip().splitlines()]
            else:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            for idx, x in enumerate(lb):
                if not seg:
                    if crop_dir is None:
                        img, _, _ = xywh2xyxy(x, w, h, img, cats=cats)
                    else:
                        img, img_crop, _ = xywh2xyxy(x, w, h, img, cats=cats, crop=True)
                        cat = cats[int(float(x[0]))]
                        save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx).replace('.png', '_%d.jpg'%idx))
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                            cv2.imwrite(save_path, img_crop)
                        else:
                            print(img_crop.shape, save_path)
                else:
                    if len(x) <= 5:
                        continue
                    if crop_dir is None:
                        img, _, _ = xywh2poly(x, w, h, img, cats=cats)
                    else:
                        img, img_crop, _ = xywh2poly(x, w, h, img, cats=cats, crop=True)
                        cat = cats[int(float(x[0]))]
                        save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx).replace('.png', '_%d.jpg'%idx))
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        if img_crop.shape[0]>0 and img_crop.shape[1]>0:
                            cv2.imwrite(save_path, img_crop)
                        else:
                            print(img_crop.shape, save_path)
            save_path = image_path.replace(img_folder, output_folder)
            cv2.imwrite(save_path, img)

def yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, crop_dir=None, attribute_file=None, filter_no=False, seg=False):
    cats = get_cats(class_file)
    if attribute_file is not None:
        with open(attribute_file, 'r') as file:
            attribute_dict = yaml.safe_load(file)['attributes']
    else:
        attribute_dict = None
    img_list = os.listdir(img_folder)
    img_list.sort()
    label_list = os.listdir(label_folder)
    label_list.sort()

    # img_list = img_list[:2]

    os.makedirs(output_folder, exist_ok=True)
    if crop_dir is not None and not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
        for cat in cats.values():
            os.makedirs(os.path.join(crop_dir, cat))

    for i in tqdm(range(len(img_list))):
        image_path = os.path.join(img_folder, img_list[i])
        label_path = os.path.join(label_folder, label_list[i])
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            if seg:
                lb = [x.split() for x in f.read().strip().splitlines()]
            else:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            for idx, x in enumerate(lb):
                attribute_len = int(x[1])
                gt_attribute = x[2:2+attribute_len]
                attribute = get_attribute(attribute_dict, gt_attribute)
                x = np.concatenate([x[:1], x[2+attribute_len:]])
                if not seg:
                    if crop_dir is None:
                        img, _, _ = xywh2xyxy(x, w, h, img, cats=cats, attributes=attribute, filter_no=filter_no)
                    else:
                        img, img_crop, attribute_strs = xywh2xyxy(x, w, h, img, cats=cats, crop=True, attributes=attribute, filter_no=filter_no)
                        cat = cats[int(x[0])]

                        save_path = os.path.join(crop_dir, cat, 'all',
                                                 os.path.basename(image_path).replace('.jpg', '_%d.jpg' % idx).replace(
                                                     '.png', '_%d.jpg' % idx))
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, img_crop)

                        for attribute_str in attribute_strs:
                            save_path = os.path.join(crop_dir, cat, attribute_str,
                                                     os.path.basename(image_path).replace('.jpg', '_%d.jpg' % idx).replace(
                                                         '.png', '_%d.jpg' % idx))
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            cv2.imwrite(save_path, img_crop)
                else:
                    if crop_dir is None:
                        img, _, _ = xywh2poly(x, w, h, img, cats=cats, attributes=attribute, filter_no=filter_no)
                    else:
                        img, img_crop, attribute_strs = xywh2poly(x, w, h, img, cats=cats, crop=True, attributes=attribute, filter_no=filter_no)
                        cat = cats[int(x[0])]
                        save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx).replace('.png', '_%d.jpg'%idx))
                        if img_crop.shape[0]>0 and img_crop.shape[1]>0:
                            cv2.imwrite(save_path, img_crop)
                        else:
                            print(img_crop.shape, save_path)
            save_path = image_path.replace(img_folder, output_folder)
            cv2.imwrite(save_path, img)


if __name__ == '__main__':
    pass
    # root_dir = r'E:\data\0318_fireservice\data0327slice'
    # root_dir = r'E:\data\0417_signboard\data0417\yolo'
    # root_dir = r'E:\data\0417_signboard\data0417\demo'
    # root_dir = r'E:\data\1123_thermal\thermal data\datasets\moisture\det'
    # root_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6'
    # root_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6_det'
    # root_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation1'
    # root_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10'
    # root_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation2'
    # root_dir = r'E:\data\tp\multi_modal_airplane_train\demo'
    # root_dir = r'E:\data\0417_signboard\data0806\dataset\yolo_rgb_detection5_10'
    # root_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10'
    # root_dir = r'E:\data\tp\sar_det'
    root_dir = r'E:\data\0111_testdata\data_new\yolo_src'
    img_folder = os.path.join(root_dir, 'images')
    label_folder = os.path.join(root_dir, 'labels')
    output_folder = os.path.join(root_dir, 'images_vis')
    crop_folder = os.path.join(root_dir, 'images_crop')
    attribute_file = os.path.join(root_dir, 'attribute.yaml')
    class_file = os.path.join(root_dir, 'class.txt')


    yolo_data_vis(img_folder, label_folder, output_folder, class_file, crop_dir=crop_folder, seg=False)
    # yolo_data_vis(img_folder, label_folder, output_folder, class_file, crop_dir=crop_folder, seg=True)
    # yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, crop_dir=crop_folder, seg=False, attribute_file=attribute_file, filter_no=True)
    # yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, crop_dir=crop_folder, seg=True, attribute_file=attribute_file, filter_no=True)