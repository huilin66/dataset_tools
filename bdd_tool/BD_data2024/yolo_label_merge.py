import os
import glob
import shutil

import pandas as pd
from tqdm import tqdm
from pathlib import Path

def yolo_txt_to_csv(label_folder, output_csv):
    """
    将YOLO格式的txt标签文件合并为一个CSV文件

    参数:
        txt_folder: 包含YOLO txt文件的文件夹路径
        output_csv: 输出的CSV文件路径
    """
    # 获取所有txt文件
    label_list = os.listdir(label_folder)

    # 存储所有数据的列表
    label_dfs = []

    names = ['class_id', 'x_center', 'y_center', 'width', 'height']

    # 遍历每个txt文件
    for label_name in tqdm(label_list):
        # 获取图像名称（不带扩展名）
        image_name_stem = Path(label_name).stem
        label_path = os.path.join(label_folder, label_name)

        df = pd.read_csv(label_path, header=None, names=names, index_col=None, sep=' ')
        df['object_id'] = df.index
        df['image_name'] = image_name_stem

        label_dfs.append(df)

    # 转换为DataFrame并保存为CSV
    df = pd.concat(label_dfs)
    df.to_csv(output_csv, index=False, sep=' ')

def get_class_list(class_file):
    df_class = pd.read_csv(class_file, index_col=None, header=None, names=['class_name'])
    class_list = df_class['class_name'].to_list()
    return class_list
def get_img_stem2name(img_dir):
    img_list = os.listdir(img_dir)
    img_stem_list = [Path(img_name).stem for img_name in img_list]
    img_stem2name_dict = dict(zip(img_stem_list, img_list))
    return img_stem2name_dict

def select_img_by_cat(csv_path, input_dir, output_dir, class_file, select_num=30, random_seed=1010):
    img_stem2name = get_img_stem2name(input_dir)
    df = pd.read_csv(csv_path, header=0, index_col=None, sep=' ')
    class_list = get_class_list(class_file)
    for class_id, class_name in enumerate(class_list):
        df_class = df[df['class_id'] == class_id]
        if df_class.shape[0]==0:
            continue
        df_class_sample = df_class.sample(n=select_num, random_state=random_seed)
        img_class_list = df_class_sample['image_name'].to_list()
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        for img_name_stem in img_class_list:
            img_name = img_stem2name[img_name_stem]
            img_path_src = os.path.join(input_dir, img_name)
            img_path_dst = os.path.join(output_class_dir, img_name)
            shutil.copy2(img_path_src, img_path_dst)

# 使用示例
if __name__ == "__main__":
    # 设置输入文件夹和输出CSV路径
    root_dir = '/data/huilin/data/BDD/cubit-det'
    image_dir = os.path.join(root_dir, 'images')
    image_select_dir = os.path.join(root_dir, 'images_select')
    label_folder = os.path.join(root_dir, 'labels')
    output_csv = os.path.join(root_dir, 'labels_merge.csv')
    class_file = os.path.join(root_dir, 'class.txt')
    yolo_txt_to_csv(label_folder, output_csv)

    select_img_by_cat(csv_path=output_csv, input_dir=image_dir, output_dir=image_select_dir, class_file=class_file)