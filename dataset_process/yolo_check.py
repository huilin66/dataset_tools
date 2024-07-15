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
if __name__ == '__main__':
    yolo_classmerge(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6\labels')