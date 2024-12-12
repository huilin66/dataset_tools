import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from skimage import io

def img_process(img):
    pass
    img_crop = img[-1280:]
    img_crop1 = img_crop[:, :1280]
    img_crop2 = img_crop[:, 1280:-1280]
    img_crop2 = cv2.resize(img_crop2, (1280, 1280))
    img_crop3 = img_crop[:, -1280:]
    return img_crop1, img_crop2, img_crop3


def img_gap_copy(source_folder, destination_folder, gap_num=3):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    file_list = os.listdir(source_folder)

    # 每10个文件选择1个进行复制
    for file_name in tqdm(file_list[::gap_num]):
        source_path = os.path.join(source_folder, file_name)
        # destination_path = os.path.join(destination_folder, file_name)
        img = io.imread(source_path)
        img_processed_list = img_process(img)
        for i, img_processed in enumerate(img_processed_list):
            destination_path = os.path.join(destination_folder, '%d'%i, file_name)
            io.imsave(destination_path, img_processed)


def yolo_copy(input_dir_list, output_dir):
    for i, input_dir in enumerate(input_dir_list):
        input_img_dir = os.path.join(input_dir, 'images')
        input_label_dir = os.path.join(input_dir, 'labels')
        output_img_dir = os.path.join(output_dir, 'images')
        output_label_dir = os.path.join(output_dir, 'labels')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        for file_name in tqdm(os.listdir(input_img_dir)):
            input_img_path = os.path.join(input_img_dir, file_name)
            output_img_path = os.path.join(output_img_dir, file_name.replace('.JPG', '_%d.JPG'%i))
            shutil.copyfile(input_img_path, output_img_path)
        for file_name in tqdm(os.listdir(input_label_dir)):
            input_label_path = os.path.join(input_label_dir, file_name)
            output_label_path = os.path.join(output_label_dir, file_name.replace('.txt', '_%d.txt'%i))
            shutil.copyfile(input_label_path, output_label_path)


if __name__ == '__main__':
    pass
    # input_dir = r'E:\data\20241113_road_veg\data\src_data2\image'
    # output_dir = r'E:\data\20241113_road_veg\data\src_data2\selected_image'
    # img_gap_copy(input_dir, output_dir)

    input_dir_list = [
        r'E:\data\20241113_road_veg\dataset\road_veg_1115_0',
        r'E:\data\20241113_road_veg\dataset\road_veg_1115_2'
    ]
    output_dir = r'E:\data\20241113_road_veg\dataset\road_veg_1115'
    yolo_copy(input_dir_list, output_dir)