import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def random_select(img_dir,  label_dir=None, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_train.to_csv(os.path.join(save_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, 'val.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, 'train.txt'),
                                           len(val_list), os.path.join(save_dir, 'val.txt')))


if __name__ == '__main__':
    pass
    dataset_dir = r'E:\data\202502_signboard\annotation_result_merge'
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    random_select(images_dir, labels_dir)