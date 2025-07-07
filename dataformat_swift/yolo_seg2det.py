import cv2
import os
import shutil
from tqdm import tqdm
from pathlib import Path

def convert_mask_to_bbox(mask_label):
    bboxes = mask_label
    return bboxes


def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


def seg2det(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    for file_name in tqdm(input_files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        new_annotations = []
        with open(input_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                x_coords = parts[1::2]
                y_coords = parts[2::2]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                new_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

        with open(output_path, 'w') as file:
            file.writelines(new_annotations)

def mseg2seg(input_dir, output_dir):
    input_label_dir = input_dir
    output_label_dir = output_dir
    label_list = os.listdir(input_label_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                attribute_num = int(num_list[1])
                num_list_seg = num_list[:1]+num_list[1+attribute_num+1:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)


if __name__ == '__main__':
    pass

    # mseg2seg(
    #     r'E:\data\202502_signboard\annotation_result_merge\labels_update',
    #     r'E:\data\202502_signboard\annotation_result_merge\labels_update_seg',
    #     )
    # seg2det(
    #     r'E:\data\202502_signboard\annotation_result_merge\labels_update_seg',
    #     r'E:\data\202502_signboard\annotation_result_merge\labels_update_det'
    #     )
