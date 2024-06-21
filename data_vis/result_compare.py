import os, shutil

import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

def cp_val_data(input_dir, output_dir, txt_path):
    os.makedirs(output_dir, exist_ok=True)
    file_list = pd.read_csv(txt_path, header=None, index_col=None)[0].tolist()
    for file_path in tqdm(file_list):
        input_path = os.path.join(input_dir, os.path.basename(file_path))
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        shutil.copyfile(input_path, output_path)

def cat_img(pre_path, gt_path):
    pre_img = cv2.imread(pre_path)
    gt_img = cv2.imread(gt_path)
    cat_img = np.concatenate((pre_img, gt_img), axis=1)
    return cat_img

def cat_show(pre_dir, gt_dir, cat_dir):
    def pre2gt_name(pre_name):
        gt_name = pre_name
        return gt_name
    file_list = os.listdir(pre_dir)
    for file_name in tqdm(file_list):
        pre_path = os.path.join(pre_dir, file_name)
        gt_path = os.path.join(gt_dir, pre2gt_name(file_name))
        cat_path = os.path.join(cat_dir, file_name)
        img_cat = cat_img(pre_path, gt_path)
        cv2.imwrite(cat_path, img_cat)

if __name__ == '__main__':
    pass