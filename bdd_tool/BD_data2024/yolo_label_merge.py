import os
import glob
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

        df = pd.read_csv(label_path, header=None, names=names, index_col=None)
        df['object_id'] = df.index
        df['image_name'] = image_name_stem

        label_dfs.append(df)

    # 转换为DataFrame并保存为CSV
    df = pd.concat(label_dfs)
    df.to_csv(output_csv, index=False)


# 使用示例
if __name__ == "__main__":
    # 设置输入文件夹和输出CSV路径
    txt_folder = 'path/to/your/txt/files'  # 替换为你的txt文件夹路径
    output_csv = 'output_labels.csv'  # 输出CSV文件名

    yolo_txt_to_csv(txt_folder, output_csv)