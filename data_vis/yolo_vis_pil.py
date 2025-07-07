import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import pandas as pd
from PIL import Image, ImageOps

red_color_bgr = (0, 0, 255)
colormap = [
    (255, 42, 4),
    (235, 219, 11),
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
    # attribute_len = 0
    # for k, v in attributes.items():
    #     attribute_len += len(v)-1
    attribute_len = len(attributes)
    return attribute_len

def get_attribute(attribute_dict, gt_attribute):
    attributes_recover = {}
    attribute_dict_key_list = list(attribute_dict.keys())

    for idx, gt_value in enumerate(gt_attribute):
        attribute_name = attribute_dict_key_list[idx]
        attribute_value = attribute_dict[attribute_name][int(gt_value)]
        attributes_recover[attribute_name] = attribute_value
    return attributes_recover
# endregion

from plotting import Annotator, colors
from copy import deepcopy
from pathlib import Path
def xywh2xyxy(x, w1, h1):
    x, y, w, h = x

    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

def xywh2poly(x, w, h, img, img_vis, cats, crop=True, attributes=None, filter_no=False, alpha=0.5, tf=1, sf=2/3):
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


    cv2.polylines(img_vis, [polys], isClosed=True, color=colormap[int(label)], thickness=2)
    mask = img_vis.copy()
    cv2.fillPoly(mask, [polys], color=colormap[int(label)])
    cv2.addWeighted(mask, alpha, img_vis, 1 - alpha, 0, img_vis)

    text_size = cv2.getTextSize(cats[int(label)], cv2.FONT_HERSHEY_SIMPLEX  , sf - 0.1, tf)[0]
    cv2.rectangle(img_vis,
                  (int(top_left_x), int(top_left_y)+10),
                  (int(top_left_x)+text_size[0]-15, int(top_left_y)+7-text_size[1]+3),
                  color=colormap[int(label)], thickness=-1)
    cv2.putText(img_vis, cats[int(label)], (int(top_left_x), int(top_left_y)+7), cv2.FONT_HERSHEY_SIMPLEX  , 0.5, (0, 0, 0), 1)

    # print(cats[int(label)], int(top_left_y),int(bottom_right_y), int(top_left_x),int(bottom_right_x), int(bottom_right_y)-int(top_left_y), int(bottom_right_x)-int(top_left_x))
    if attributes is not None:
        count = 0
        count2 = 0
        attribute_strs = []

        br_poss = []
        for idx, (k, v) in enumerate(attributes.items()):
            text = f'{k}-{v}'
            if filter_no:
                if not v or v == 'no':
                    continue
            count += 1
            color = (255, 0, 0) if v is not False else (0, 0, 0)
            cv2.putText(img_vis, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count), cv2.FONT_HERSHEY_SIMPLEX  , 0.5, color, 1)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX  , 0.5, 1)
            attribute_strs.append(text)
            br_poss.append([int(top_left_x)+text_width-15, int(top_left_y) + 12+10*count])

        if len(br_poss)>0:
            br_poss = np.array(br_poss)
            tl_pos = [int(top_left_x), int(top_left_y)+10]
            br_pos = [np.max(br_poss, axis=0)[0], br_poss[-1][1]]
            box = tl_pos + br_pos
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2])+15, int(box[3]+2))
            cv2.rectangle(img_vis, p1, p2, (255, 255, 255))
            overlay = img_vis.copy()
            cv2.rectangle(overlay, p1, p2, (255, 255, 255), -1)
            cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0, img_vis)

            br_poss = []
            for idx, (k, v) in enumerate(attributes.items()):
                text = f'{k}-{v}'
                if filter_no:
                    if not v or v == 'no':
                        continue
                count2 += 1
                color = (255, 0, 0) if v is not False else (0, 0, 0)
                # text_size = cv2.putText(img, text, (int(top_left_x), int(top_left_y) + 12+10*count), cv2.FONT_HERSHEY_SIMPLEX  , 0.5, color, 1)[0]
                cv2.putText(img_vis, text, (int(top_left_x), int(top_left_y) + 12 + 10 * count2), cv2.FONT_HERSHEY_SIMPLEX  ,
                            0.5, color, 1)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX  , 0.5, 1)
                attribute_strs.append(text)
                br_poss.append([int(top_left_x) + text_width - 15, int(top_left_y) + 12 + 10 * count2])
    else:
        attribute_strs = None

    return img_vis, img_crop, attribute_strs

def yolo_data_vis(img_folder, label_folder, output_folder, class_file, crop_dir=None, seg=False):
    cats = get_cats(class_file)
    img_list = os.listdir(img_folder)
    img_list.sort()
    label_list = os.listdir(label_folder)
    label_list.sort()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if crop_dir is not None and not os.path.exists(crop_dir):
        os.makedirs(crop_dir, exist_ok=True)
        for cat in cats.values():
            os.makedirs(os.path.join(crop_dir, cat), exist_ok=True)

    for i in tqdm(range(len(img_list))):
        image_path = os.path.join(img_folder, img_list[i])
        label_path = os.path.join(label_folder, label_list[i])
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        img_vis = img.copy()
        with open(label_path, 'r') as f:
            if seg:
                lb = [x.split() for x in f.read().strip().splitlines()]
            else:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            for idx, x in enumerate(lb):
                if not seg:
                    if crop_dir is None:
                        img_vis, _, _ = xywh2xyxy(x, w, h, img, img_vis, cats=cats)
                    # else:
                    #     img_vis, img_crop, _ = xywh2xyxy(x, w, h, img, img_vis, cats=cats, crop=True)
                    #     cat = cats[int(float(x[0]))]
                    #     save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx).replace('.png', '_%d.jpg'%idx))
                    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    #     if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                    #         cv2.imwrite(save_path, img_crop)
                    #     else:
                    #         print(img_crop.shape, save_path)
                # else:
                #     if len(x) <= 5:
                #         continue
                #     if crop_dir is None:
                #         img_vis, _, _ = xywh2poly(x, w, h, img, img_vis, img_viscats=cats)
                #     else:
                #         img_vis, img_crop, _ = xywh2poly(x, w, h, img, img_vis, cats=cats, crop=True)
                #         cat = cats[int(float(x[0]))]
                #         save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx).replace('.png', '_%d.jpg'%idx))
                #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
                #         if img_crop.shape[0]>0 and img_crop.shape[1]>0:
                #             cv2.imwrite(save_path, img_crop)
                #         else:
                #             print(img_crop.shape, save_path)
            save_path = image_path.replace(img_folder, output_folder)
            cv2.imwrite(save_path, img_vis)

def yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, crop_dir=None, attribute_file=None,
                  filter_no=False, seg=False, crop_keep_shape=False, det_crop=True, pred=False, suffix='', with_conf=True):

    cats = get_cats(class_file)
    if attribute_file is not None:
        with open(attribute_file, 'r') as file:
            attribute_dict = yaml.load(file, Loader=yaml.BaseLoader)['attributes']
    else:
        attribute_dict = None
    img_list = os.listdir(img_folder)
    img_list.sort()
    label_list = os.listdir(label_folder)
    label_list.sort()
    assert len(img_list) == len(label_list), print('the number of images and labels do not match')
    # img_list = img_list[:2]

    os.makedirs(output_folder, exist_ok=True)
    print(output_folder)
    if crop_dir is not None and not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
        for cat in cats.values():
            os.makedirs(os.path.join(crop_dir, cat))

    for i in tqdm(range(len(img_list))):
        image_path = os.path.join(img_folder, img_list[i])
        label_path = os.path.join(label_folder, label_list[i])
        save_path = os.path.join(output_folder, img_list[i].replace('.png', suffix+'.jpg'))
        img = Image.open(image_path)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            if seg:
                lb = [x.split() for x in f.read().strip().splitlines()]
            else:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)

            annotator = Annotator(
                deepcopy(img),
                line_width=None,
                font_size=None,
                font="Arial.tff",
                pil=False,  # Classify tasks default to pil=True
                example=cats,
            )

            for idx, x in enumerate(lb):
                c = int(x[0])
                attribute_len = int(x[1])
                gt_attribute = x[2:2+attribute_len]
                box_xywh = x[2+attribute_len: 2+attribute_len+4]
                if with_conf:
                    conf = float(x[-1])
                else:
                    conf = None
                if pred:
                    gt_attribute = [1 if att>0.01 else 0 for att in gt_attribute]
                attribute = get_attribute(attribute_dict, gt_attribute)
                box_xyxy= xywh2xyxy(box_xywh, w, h)

                name = cats[c]
                label = f"{name} {conf:.2f}" if with_conf else f"{name}"
                pos_base = annotator.box_label(box_xyxy, label, color=colors(c, True))

                count = 0
                br_poss = []
                for idx, (k, v) in enumerate(attribute.items()):
                    label = f'{k}-{v}'
                    if filter_no and 'no' in v:
                        continue
                    count += 1
                    pos = [pos_base[0], pos_base[1]+15*(count+1)-10]
                    br_pos = annotator.text(pos, label, txt_color=colors(idx, True))
                    br_poss.append(br_pos)
                if len(br_poss)>0:
                    br_poss = np.array(br_poss)
                    tl_pos = pos_base
                    br_pos = (np.max(br_poss, axis=0)[0], br_poss[-1][1])
                    annotator.rectangle_mask(box=tl_pos + br_pos, color=(255, 255, 255), alpha=0.5, )
                count_new = 0
                for idx, (k, v) in enumerate(attribute.items()):
                    label = f'{k}-{v}'
                    if filter_no and 'no' in v:
                        continue
                    count_new += 1
                    pos = [pos_base[0], pos_base[1]+15*(count_new+1)-10]
                    br_pos = annotator.text(pos, label, txt_color=colors(idx, True))

            annotator.save(save_path)


if __name__ == '__main__':
    pass
    root_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c'
    # img_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images'
    img_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction_\image'
    output_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction_\all2'
    attribute_file = os.path.join(root_dir, 'attribute.yaml')
    class_file = os.path.join(root_dir, 'class.txt')

    label_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction2_\prediction_yolo8x2\labels'
    yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, attribute_file=attribute_file, filter_no=True, pred=True, suffix='_yolo8x')

    label_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction2_\prediction_yolo9e2\labels'
    yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, attribute_file=attribute_file, filter_no=True, pred=True, suffix='_yolo9e')

    label_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction2_\prediction_yolo10x2\labels'
    yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, attribute_file=attribute_file, filter_no=True, pred=True, suffix='_yolo10x')

    label_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\prediction2_\prediction_mayolox2\labels'
    yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, attribute_file=attribute_file, filter_no=True, pred=True, suffix='_mayolo')

    # label_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4\labels'
    # output_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4_infer_vis'
    # yolo_mdet_vis(img_folder, label_folder, output_folder, class_file, attribute_file=attribute_file, filter_no=True,
    #               pred=True, suffix='', with_conf=False)
