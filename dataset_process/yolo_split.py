import os
import numpy as np
import pandas as pd

def random_select(img_dir, dst_dir, train_ratio=0.9, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_train.to_csv(os.path.join(dst_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(dst_dir, 'val.txt'), header=None, index=None)

def gap_select(img_dir, dst_dir, train_ratio=0.9, full_path=True):
    file_list = os.listdir(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]

    interval = len(file_list) // int((1-train_ratio)*100)  # 计算间隔

    val_list = file_list[::interval]  # 使用切片功能取出等间隔的元素
    train_list = [item for i, item in enumerate(file_list) if i % interval != 0]  # 将其余元素保存到train_list

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_train.to_csv(os.path.join(dst_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(dst_dir, 'val.txt'), header=None, index=None)


if __name__ == '__main__':
    pass
    # random_select(r'E:\data\0318_fireservice\data0318\images',
    #               r'E:\data\0318_fireservice\data0318')
    gap_select(r'E:\data\0318_fireservice\data0318\images',
                  r'E:\data\0318_fireservice\data0318')