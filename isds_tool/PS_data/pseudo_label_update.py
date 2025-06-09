import os
import shutil
from pathlib import Path
from tqdm import tqdm


def mseg_class_update(input_label_dir, output_label_dir):
    label_list = os.listdir(input_label_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                lines[idx] = str(int(lines[idx][0])+1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    pass
    input_label_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\images_crop_box_infer5_updated'
    output_label_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\labels_updated'
    mseg_class_update(input_label_dir, output_label_dir)