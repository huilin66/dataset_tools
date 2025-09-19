import os
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def get_img_list(csv_path):
    pass
    df = pd.read_csv(csv_path)
    img_list = df['file_name'].to_list()
    return img_list

def data_cp(input_dir, output_dir, img_list):
    input_img_dir = os.path.join(input_dir, 'images')
    output_img_dir = os.path.join(output_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        label_name = Path(img_name).with_suffix('.txt')
        input_img_path = os.path.join(input_img_dir, img_name)
        output_img_path = os.path.join(output_img_dir, img_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        shutil.copy(input_img_path, output_img_path)
        shutil.copy(input_label_path, output_label_path)

if __name__ == '__main__':
    csv_path = r'/localnvme/data/billboard/check0914_split.csv'
    input_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0912'
    output_dir = r'/localnvme/data/billboard/check_0914'
    img_list = get_img_list(csv_path)
    data_cp(input_dir, output_dir, img_list)