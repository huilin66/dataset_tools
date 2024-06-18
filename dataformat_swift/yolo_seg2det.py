import cv2
import os
from tqdm import tqdm

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


def convert_instance_segmentation_to_detection(input_dir, output_dir):
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


if __name__ == '__main__':
    # Example usage:
    convert_instance_segmentation_to_detection(
        r'E:\data\1123_thermal\thermal data\datasets\moisture\seg\labels',
        r'E:\data\1123_thermal\thermal data\datasets\moisture\det\labels',
        )
