import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

class_list = [0, 1]

def yolo_classmerge(input_dir):
    '''

    :param input_dir:
    :param copy_dir:
    :return:
    '''

    file_list = os.listdir(input_dir)

    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)


        with open(input_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            if numbers[0] in class_list:
                continue
            else:
                print(file_name, numbers)

def check_mdet(label_dir, len=10):
    label_list = os.listdir(label_dir)
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        print(label_name)
        df = pd.read_csv(label_path, header=None, index_col=None, sep=' ')
        print(df.shape)

if __name__ == '__main__':
    # yolo_classmerge(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6\labels')
    check_mdet(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels')