import os
import shutil

from tqdm import tqdm
from pathlib import Path


def dataset_merge(root_dir, dst_dir):
    dst_image_dir = os.path.join(dst_dir, 'images')
    dst_label_dir = os.path.join(dst_dir, 'labels')
    os.makedirs(dst_image_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    dataset_list = os.listdir(root_dir)
    for idx, dataset_name in enumerate(dataset_list):
        dataset_dir = os.path.join(root_dir, dataset_name)
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        images_list = os.listdir(images_dir)
        for image_name in tqdm(images_list, desc=f'merge {idx}/{len(dataset_list)}'):
            label_name = Path(image_name).stem + '.txt'
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                continue
            dst_image_path = os.path.join(dst_image_dir, image_name)
            dst_label_path = os.path.join(dst_label_dir, label_name)
            shutil.copy(image_path, dst_image_path)
            shutil.copy(label_path, dst_label_path)


if __name__ == '__main__':
    pass
    root_dir = r'E:\data\202502_signboard\annotation_result'
    dst_dir = r'E:\data\202502_signboard\annotation_result_merge'
    dataset_merge(root_dir, dst_dir)