import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class_dict = {
    0 : 'background',
    1 : 'wall_signboard',
    2: 'projecting_signboard',
}
def random_select(img_dir, dst_dir, label_dir=None, train_ratio=0.9, random_seed=1010, full_path=True):
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
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(dst_dir, 'train.txt'),
                                           len(val_list), os.path.join(dst_dir, 'val.txt')))

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

def get_class(class_dict, class_path):
    df_train = pd.DataFrame({'cat_name': list(class_dict.values())})
    df_train.to_csv(class_path, header=None, index=None)


def copy_split(img_dir, gt_dir, dst_img_dir, dst_gt_dir, ref_path):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_gt_dir, exist_ok=True)
    df = pd.read_csv(ref_path, header=None, index_col=None, names=['file_path'])
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['file_path']
        gt_path = img_path.replace(img_dir, gt_dir).replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.png', '.txt')
        dst_img_path = img_path.replace(img_dir, dst_img_dir)
        dst_gt_path = gt_path.replace(gt_dir, dst_gt_dir)

        shutil.copyfile(img_path, dst_img_path)
        shutil.copyfile(gt_path, dst_gt_path)

if __name__ == '__main__':
    pass
    # random_select(r'E:\data\0318_fireservice\data0318\images',
    #               r'E:\data\0318_fireservice\data0318')
    # gap_select(r'E:\data\0318_fireservice\data0318\images',
    #               r'E:\data\0318_fireservice\data0318')

    # random_select(r'E:\data\0416_trafficsign\GTSDB\images',
    #               r'E:\data\0416_trafficsign\GTSDB')
    # get_class(class_dict, r'E:\data\0416_trafficsign\GTSDB\classes.txt')


    # random_select(r'E:\data\0417_signboard\data0521_m\yolo\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo')

    # random_select(r'E:\data\1123_thermal\thermal data\datasets\moisture\images',
    #               r'E:\data\1123_thermal\thermal data\datasets\moisture')


    # random_select(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5', train_ratio=0.907)

    # random_select(r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation1\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation1', train_ratio=0.9)

    # random_select(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_f\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_f', train_ratio=0.9)

    # random_select(r'E:\data\tp\multi_modal_airplane_train\images',
    #               r'E:\data\tp\multi_modal_airplane_train', train_ratio=0.9)

    # random_select(r'E:\data\tp\car_det_train\car_det_train\img',
    #               r'E:\data\tp\car_det_train\car_det_train', train_ratio=0.9)

    # random_select(r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation2\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo_rgb_segmentation2', train_ratio=0.9)

    # random_select(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\images',
    #               r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10', train_ratio=0.9)

    # random_select(r'E:\data\tp\sar_det\images',
    #               r'E:\data\tp\sar_det', train_ratio=0.9)
    # random_select(r'E:\data\0111_testdata\data_new\yolo_src\images',
    #               r'E:\data\0111_testdata\data_new\yolo_src', train_ratio=0.9)

    # copy_split(
    #     r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\images',
    #     r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\labels',
    #     r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\images_val',
    #     r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\labels_val',
    #     r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\val.txt',
    # )

    # copy_split(
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images',
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images_val',
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels_val',
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\val.txt',
    # )

    # random_select(r'E:\data\202411_trafficsign\traff_sign_yolo\images',
    #               r'E:\data\202411_trafficsign\traff_sign_yolo',
    #               r'E:\data\202411_trafficsign\traff_sign_yolo\labels',
    #               train_ratio=0.9)


    copy_split(
        r'E:\data\202411_trafficsign\traff_sign_yolo\images',
        r'E:\data\202411_trafficsign\traff_sign_yolo\labels',
        r'E:\data\202411_trafficsign\traff_sign_yolo\images_val',
        r'E:\data\202411_trafficsign\traff_sign_yolo\labels_val',
        r'E:\data\202411_trafficsign\traff_sign_yolo\val.txt',
    )