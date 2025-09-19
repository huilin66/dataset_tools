import os
import shutil

import pandas as pd
from tqdm import tqdm
from pathlib import Path

def replace_last(s, old, new):
    parts = s.rsplit(old, 1)
    return new.join(parts) if len(parts)>1 else s

def get_object_list(input_path, stem2image):
    df = pd.read_excel(input_path, sheet_name='Sheet1')
    print(df)

    image_list = []
    for idx, row in df.iterrows():
        object_name = row['object_name']
        suffix = '_'+object_name.split('_')[-1]
        object_stem = replace_last(object_name, suffix, '')
        image_name = stem2image[object_stem]
        df.loc[idx, 'image_name'] = image_name

        image_list.append(image_name)
    df.to_excel(input_path, sheet_name='Sheet1', index=False)
    return image_list
def get_stem2image(input_dir):
    image_list = os.listdir(input_dir)
    stem_list = [Path(image_name).stem for image_name in image_list]
    stem2image = dict(zip(stem_list, image_list))
    return stem2image

def copy_data(input_data, output_data, image_list):
    input_image = os.path.join(input_data, 'images')
    input_label = os.path.join(input_data, 'labels')
    output_image = os.path.join(output_data, 'images')
    output_label = os.path.join(output_data, 'labels')
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).with_suffix('.txt')
        input_image_path = os.path.join(input_image, image_name)
        input_label_path = os.path.join(input_label, label_name)
        output_image_path = os.path.join(output_image, image_name)
        output_label_path = os.path.join(output_label, label_name)
        shutil.copyfile(input_image_path, output_image_path)
        shutil.copyfile(input_label_path, output_label_path)

if __name__ == '__main__':
    pass
    input_path = r'/localnvme/data/billboard/image_list2.xlsx'
    input_data = r'/localnvme/data/billboard/fused_data/data6010_mseg_c6_0903'
    input_image = os.path.join(input_data, 'images')
    input_label = os.path.join(input_data, 'labels')
    output_data = r'/localnvme/data/billboard/select_data0905'
    stem2image = get_stem2image(input_image)
    image_list = get_object_list(input_path, stem2image)
    copy_data(input_data,output_data, image_list)