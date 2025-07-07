import shutil

import pandas as pd
import yaml
import os
from skimage import io
from pathlib import Path
from tqdm import tqdm

unique_labels = ['off', 'Yellow',
                 'Red', 'RedLeft', 'RedRight', 'RedStraight', 'RedStraightLeft',
                 'Green', 'GreenLeft', 'GreenRight', 'GreenStraight', 'GreenStraightLeft', 'GreenStraightRight',]

def get_unique_labels(yaml_path):
    unique_labels = []
    with open(yaml_path, "r") as f:
        annotations = yaml.safe_load(f)
    for item in annotations:
        boxes = item["boxes"]
        for box in boxes:
            label = box["label"]
            if label not in unique_labels:
                unique_labels.append(label)
    print(unique_labels, f'in {os.path.basename(yaml_path)}')

def get_class4(label):
    if 'off' in label:
        return 0
    elif 'Yellow' in label or 'yellow' in label:
        return 1
    elif 'Red' in label or 'red' in label:
        return 2
    elif 'Green' in label or 'green' in label:
        return 3
    else:
        print(label)
        return 4

def bosch2yolo(yaml_path, yolo_label_dir, image_dir, yolo_img_dir, test_format=False, keep_empty=False):
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_label_dir, exist_ok=True)

    with open(yaml_path, "r") as f:
        annotations = yaml.safe_load(f)
    for item in tqdm(annotations):
        boxes = item["boxes"]
        if not test_format:
            img_path = os.path.join(image_dir, item["path"])
        else:
            img_path = os.path.join(image_dir, 'rgb', 'test', os.path.basename(item["path"]))

        yolo_img_path = os.path.join(yolo_img_dir, os.path.basename(img_path))
        yolo_label_path = os.path.join(yolo_label_dir, Path(img_path).stem + '.txt')
        df = pd.DataFrame(None, columns=['label', 'x', 'y', 'h', 'w'])
        img = io.imread(img_path)
        img_height, img_width = img.shape[:2]
        if len(boxes) == 0 and not keep_empty:
            continue
        for box in boxes:
            label = get_class4(box["label"])
            x_min, x_max = box["x_min"], box["x_max"]
            y_min, y_max = box["y_min"], box["y_max"]

            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            df.loc[len(df)] = [label, x_center, y_center, width, height]
        df.to_csv(yolo_label_path, index=False, header=False, sep=' ')
        shutil.copy(img_path, yolo_img_path)


if __name__ == "__main__":
    train_add_path = r'E:\data\202411_trafficsign\Bosch Small Traffic Lights Dataset\dataset_additional_rgb\additional_train.yaml'
    train_path = r'E:\data\202411_trafficsign\Bosch Small Traffic Lights Dataset\dataset_train_rgb\dataset_train_rgb\train.yaml'
    test_path = r'E:\data\202411_trafficsign\Bosch Small Traffic Lights Dataset\dataset_test_rgb\dataset_test_rgb\test.yaml'
    yolo_img_dir = r'E:\data\202411_trafficsign\traff_sign_yolo\images'
    yolo_label_dir = r'E:\data\202411_trafficsign\traff_sign_yolo\labels'
    # get_unique_labels(train_add_path)
    # get_unique_labels(train_path)
    # get_unique_labels(test_path)
    bosch2yolo(train_add_path, yolo_label_dir, os.path.dirname(train_add_path), yolo_img_dir) #215
    bosch2yolo(train_path, yolo_label_dir, os.path.dirname(train_path), yolo_img_dir) # 5093
    bosch2yolo(test_path, yolo_label_dir, os.path.dirname(test_path), yolo_img_dir, test_format=True) # 8334