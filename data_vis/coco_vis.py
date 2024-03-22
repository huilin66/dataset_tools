import json
import os
import cv2
import numpy as np
from tqdm import tqdm


# COCO数据集的json文件路径和图片路径
train_json = r'E:\Huilin\2308_concretespalling\data\mk_merge\coco_select\annotations\instance_val.json'
train_path = r'E:\Huilin\2308_concretespalling\data\mk_merge\coco_select\img'
vis_path = r'E:\Huilin\2308_concretespalling\data\mk_merge\yolo_select\random\images\val_vis'

rtdetr_path = r'E:\Huilin\2308_concretespalling\data\mk_merge\yolo_select\random\images\val_rtdetr'
compare_path = r'E:\Huilin\2308_concretespalling\data\mk_merge\yolo_select\random\images\val_compare'

def visualization_bbox(json_path, img_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(json_path) as annos:
        annotation_json = json.load(annos)


    for image_record in tqdm(annotation_json['images']):
        image_name = image_record['file_name']
        image_path = os.path.join(img_dir, image_name)
        image = cv2.imread(image_path, 1)

        num_bbox = 0
        for i in range(len(annotation_json['annotations'][::])):
            if annotation_json['annotations'][i - 1]['image_id'] == id:
                num_bbox = num_bbox + 1
                x, y, w, h = annotation_json['annotations'][i - 1]['bbox']
                image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

        cv2.imwrite(os.path.join(dst_dir, image_name), image)


def dir_compare(dir1, dir2, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    file_list = os.listdir(dir1)
    for file_name in tqdm(file_list):
        file_path1 = os.path.join(dir1, file_name)
        file_path2 = os.path.join(dir2, file_name)
        file_path_dst = os.path.join(dst_dir, file_name)

        img1 = cv2.imread(file_path1)
        img2 = cv2.imread(file_path2)

        img = np.hstack((img1, img2))

        cv2.imwrite(file_path_dst, img)

if __name__ == "__main__":
    # visualization_bbox(train_json, train_path, vis_path)

    dir_compare(vis_path, rtdetr_path, compare_path)