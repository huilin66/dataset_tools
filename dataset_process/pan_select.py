import os
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd


def img_select(img_dir, dst_dir):
    pass
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    file_list = os.listdir(img_dir)
    file_list_select = file_list[::20]
    for file_name in tqdm(file_list_select):
        input_path = os.path.join(img_dir, file_name)
        output_path = os.path.join(dst_dir, file_name)
        shutil.copy(input_path, output_path)

def rm_nogt(img_dir):
    file_list = os.listdir(img_dir)
    img_list = [os.path.join(img_dir, file_name) for file_name in file_list if file_name.endswith('.jpg')]
    for img_path in tqdm(img_list):
        gt_path = img_path.replace('.jpg', '.json')
        if not os.path.exists(gt_path):
            os.remove(img_path)


def data_merge(data_dir_list, dst_dir):
    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir)
    for src_dir in data_dir_list:
        for file_name in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file_name)
            dst_path = file_path.replace(src_dir, dst_dir)
            shutil.copy(file_path, dst_path)


def val_select(img_dir_all, gt_dir_all,
               txt_train_path, txt_val_path,
               random=False, select_ratio=0.1, random_seed=1010):
    img_list = os.listdir(img_dir_all)
    if random:
        np.random.seed(random_seed)
        np.random.shuffle(img_list)

    interval = int(len(img_list)//(len(img_list)*select_ratio))  # 计算间隔
    img_list_val = img_list[::interval]  # 使用切片功能取出等间隔的元素
    img_list_train = [item for i, item in enumerate(img_list) if i % interval != 0]  # 将其余元素保存到train_list
    img_list_val = [os.path.join(img_dir_all, file_name) for file_name in img_list_val]
    img_list_train = [os.path.join(img_dir_all, file_name) for file_name in img_list_train]

    # gt_list_val = [img_path.replace(img_dir_all, gt_dir_all).replace('.jpg', '.json') for img_path in img_list_val]
    # gt_list_train = [img_path.replace(img_dir_all, gt_dir_all).replace('.jpg', '.json') for img_path in img_list_train]

    df_train = pd.DataFrame({'filename': img_list_train})
    df_train.to_csv(txt_train_path, header=None, index=None)
    df_val = pd.DataFrame({'filename': img_list_val})
    df_val.to_csv(txt_val_path, header=None, index=None)

    # for img_name in tqdm(img_list_val):
    #     img_path = os.path.join(img_dir_all, img_name)
    #     img_select_path = img_path.replace(img_dir_all, img_dir_select)
    #     gt_path = img_path.replace(img_dir_all, gt_dir_all).replace('.png', '.txt')
    #     gt_select_path = gt_path.replace(gt_dir_all, gt_dir_select)
    #     shutil.move(img_path, img_select_path)
    #     shutil.move(gt_path, gt_select_path)



if __name__ == '__main__':
    pass
    # merge_dir = r'E:\data\0417_signboard\data0417'
    merge_dir = r'E:\data\0417_signboard\data0420\labelme'
    data_list = [
        'mk01',
        # 'mk02',
        # 'mk03',
        # 'mk04',
        # 'mk05',
        # 'mk06',
        'mk07',
        # 'mk18',
        'mk19',
        'mk22',
    ]
    for dataname in data_list:
        img_dir = r'E:\data\1211_monhkok\%s\imgs'%dataname
        dst_dir = r'E:\data\1211_monhkok\%s\imgs_select'%dataname
        # img_select(img_dir, dst_dir)
        rm_nogt(dst_dir)
        data_merge([dst_dir], merge_dir)

    # img_dir = r'E:\data\0417_signboard\data_yj\imgs'
    # dst_dir = r'E:\data\0417_signboard\data_yj\imgs_select'
    # img_select(img_dir, dst_dir)
    # rm_nogt(dst_dir)
    # data_merge([dst_dir], merge_dir)


    # yolo_train_dir = r'E:\data\0417_signboard\data0420\yolo\images'
    # yolo_gt_dir = r'E:\data\0417_signboard\data0420\yolo\labels'
    # txt_train_path = r'E:\data\0417_signboard\data0420\yolo\train.txt'
    # txt_val_path = r'E:\data\0417_signboard\data0420\yolo\val.txt'
    # val_select(yolo_train_dir, yolo_gt_dir,
    #            txt_train_path, txt_val_path,
    #            random=False, select_ratio=0.1)