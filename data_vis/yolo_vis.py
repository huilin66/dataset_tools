import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

'''
Automatic Sprinkler
Fire Detectors
Fire Alarm Bell
Fire Alarm_break grass
Fire Hose Reel
Exit
Fire Alarm Bell round
Fire Alarm Bell white
'''

# cats = {
#     0: 'Automatic Sprinkler',
#     1: 'Fire Detectors',
#     2: 'Fire Alarm Bell',
#     3: 'Fire Alarm_break grass',
#     4: 'Fire Hose Reel',
#     5: 'Exit',
#     6: 'Fire Alarm Bell round',
#     7: 'Fire Alarm Bell white',
#     8: 'Fire Detectors white',
#     9: 'Fire Alarm Bell flat',
# }

# cats = {
#     0: 'background',
#     1: 'crack',
#     2: 'mold',
#     3: 'peeling_paint',
#     4: 'stairstep_crack',
#     5: 'water_seepage',
#     6: 'spall',
# }

cats = {
    0: 'boardin',
    1: 'boardout',
}

colormap = [(0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (255, 128, 0),
            (128, 255, 0),
            (128, 128, 0),
            (128, 0, 128),
            ]  # 色盘，可根据类别添加新颜色

# 坐标转换
def xywh2xyxy(x, w1, h1, img, crop=True):
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
    cv2.putText(img, cats[int(label)], (int(top_left_x), int(top_left_y) +10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img, img_crop

def xywh2poly(x, w, h, img, crop=True):
    label, polypos = int(x[0]),x[1:]
    polys = []
    for i in range(0, len(polypos), 2):
        pos1 = float(polypos[i]) * w
        pos2 = float(polypos[i+1]) * h
        polys.append([pos1, pos2])

    polys = np.array(polys, np.int32)
    if crop:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.polylines(mask, [polys], 1, 255)
        cv2.fillPoly(mask, [polys], 255)
        img_crop = cv2.bitwise_and(img, img, mask=mask)
        top_left_x,top_left_y = np.min(polys, axis=0)
        bottom_right_x, bottom_right_y = np.max(polys, axis=0)
        img_crop = img_crop[int(top_left_y):int(bottom_right_y), int(top_left_x):int(bottom_right_x)].copy()
    else:
        img_crop = None
    cv2.polylines(img, [polys], isClosed=True, color=colormap[int(label)], thickness=2)
    return img, img_crop


def yolo_data_vis(img_folder, label_folder, output_folder, crop_dir=None, seg=False):
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
                        img, _ = xywh2xyxy(x, w, h, img)
                    else:
                        img, img_crop = xywh2xyxy(x, w, h, img, crop=True)
                        cat = cats[int(x[0])]
                        save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx))
                        cv2.imwrite(save_path, img_crop)
                else:
                    if crop_dir is None:
                        img, _ = xywh2poly(x, w, h, img)
                    else:
                        img, img_crop = xywh2poly(x, w, h, img, crop=True)
                        cat = cats[int(x[0])]
                        save_path = os.path.join(crop_dir, cat, os.path.basename(image_path).replace('.jpg', '_%d.jpg'%idx))
                        cv2.imwrite(save_path, img_crop)
            save_path = image_path.replace(img_folder, output_folder)
            cv2.imwrite(save_path, img)

if __name__ == '__main__':
    pass
    # root_dir = r'E:\data\0318_fireservice\data0327slice'
    root_dir = r'E:\data\0417_signboard\data0417\yolo'
    img_folder = os.path.join(root_dir, 'images')
    label_folder = os.path.join(root_dir, 'labels')
    output_folder = os.path.join(root_dir, 'images_vis')
    crop_folder = os.path.join(root_dir, 'images_crop')
    yolo_data_vis(img_folder, label_folder, output_folder, crop_dir=crop_folder, seg=True)