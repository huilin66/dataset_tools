import os
import ast
import cv2
import yaml
import json
import shutil
import pandas as pd
import numpy as np
from skimage import io
from tqdm import tqdm
from json_split import *
from data_vis.json_vis import anno_vis
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from scipy.spatial.distance import euclidean
from data_vis.yolo_vis import yolo_data_vis
from natsort import natsorted

categories = ['background',
              'crack', 'hole', 'blister', 'delamination', 'peeling',
              'spalling', 'mold', 'corrosion', 'condensation', 'stain',
              'vegetation', 'mix']

# region yolo tools

# region tools
def shift_judge(imageA, imageB):
    if imageA.shape==imageB.shape:
        pass
    elif imageA.shape[0] / imageB.shape[0] == imageA.shape[1] / imageB.shape[1]:
        imageA = cv2.resize(imageA, [imageB.shape[1], imageB.shape[0]])
    else:
        return 0

    gray_imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    gray_imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    shift, p = cv2.phaseCorrelate(np.float32(gray_imageA), np.float32(gray_imageB))
    return p

def same_judge(imageA, imageB):
    if imageA.shape==imageB.shape:
        pass
    elif imageA.shape[0] / imageB.shape[0] == imageA.shape[1] / imageB.shape[1]:
        imageA = cv2.resize(imageA, [imageB.shape[1], imageB.shape[0]])
    else:
        return False
    gray_imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    gray_imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    similar_score = ssim(gray_imageA, gray_imageB)
    if similar_score > 0.9:
        return True
    else:
        return False

def rotate_90(image):
    return np.rot90(image, k=-1)

def rotate_180(image):
    return np.rot90(image, k=2)

def rotate_270(image):
    return np.rot90(image, k=1)

def mirror_left_right(image):
    return np.fliplr(image)

def mirror_up_down(image):
    return np.flipud(image)

def mirror_main_diagonal(image):
    return np.transpose(image, (1, 0, 2))

def mirror_secondary_diagonal(image):
    return np.rot90(np.transpose(image, (1, 0, 2)), k=2)

def rot_judge(imageA, imageB):
    if same_judge(imageA, rotate_90(imageB)):
        return True
    if same_judge(imageA, rotate_180(imageB)):
        return True
    if same_judge(imageA, rotate_270(imageB)):
        return True
    return False

def mirror_simple_judge(imageA, imageB):
    if imageA.shape==imageB.shape:
        pass
    elif imageA.shape[0] / imageB.shape[0] == imageA.shape[1] / imageB.shape[1]:
        imageA = cv2.resize(imageA, [imageB.shape[1], imageB.shape[0]])
    else:
        return False
    if same_judge(imageA, mirror_left_right(imageB)):
        return True
    if same_judge(imageA, mirror_up_down(imageB)):
        return True
    return False

def mirror_diagonal_judge(imageA, imageB):
    if same_judge(imageA, mirror_main_diagonal(imageB)):
        return True
    if same_judge(imageA, mirror_secondary_diagonal(imageB)):
        return True
    return False

def mirror_judge(imageA, imageB):
    if mirror_simple_judge(imageA, mirror_left_right(imageB)):
        return True
    if mirror_diagonal_judge(imageA, mirror_up_down(imageB)):
        return True
    return False
def is_instance_segmentation(file_path):
    """
    判断给定的 YOLO 标注文件是目标检测标注还是实例分割标注。
    如果每一行有 5 个数值，则是目标检测标注；如果有更多数值，则可能是实例分割标注。
    """
    with open(file_path, 'r') as file:
        for line in file:
            # 将每行拆分为列表，基于空格或逗号分隔
            data = line.strip().split()

            # 如果每行只有5个元素，表示目标检测标注格式 <class_id> <x_center> <y_center> <width> <height>
            if len(data) == 5:
                return False
            # 如果每行有更多数据，则可能是实例分割标注
            elif len(data) > 5:
                return True


def convert_yolo_hybrid_to_detection(input_file, output_file):
    """
    将归一化的 YOLO 实例分割标注文件转换为目标检测标注文件（Bounding Box），
    不需要对坐标再次进行归一化。

    参数:
    input_file: YOLO 实例分割标注文件的路径（已经归一化的多边形）。
    output_file: 输出的 YOLO 目标检测标注文件路径。
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 读取每行标注
            values = list(map(float, line.strip().split()))

            class_id = int(values[0])  # 类别ID
            polygon = values[1:]  # 多边形坐标 (已经归一化)
            if len(polygon) == 4:
                x_center, y_center, bbox_width, bbox_height = polygon
            else:
                # 计算多边形的最小包围框 (Bounding Box)
                x_coords = polygon[0::2]  # 所有的 x 坐标
                y_coords = polygon[1::2]  # 所有的 y 坐标

                x_min = max(min(x_coords), 0)
                x_max = min(max(x_coords), 1)
                y_min = max(min(y_coords), 0)
                y_max = min(max(y_coords), 1)

                # 计算 YOLO 格式下的边界框 (x_center, y_center, width, height)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                x_center = x_min + bbox_width / 2
                y_center = y_min + bbox_height / 2

            # 写入 YOLO 目标检测标注格式: class_id x_center y_center width height
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


def convert_yolo_segmentation_to_detection(input_file, output_file):
    """
    将归一化的 YOLO 实例分割标注文件转换为目标检测标注文件（Bounding Box），
    不需要对坐标再次进行归一化。

    参数:
    input_file: YOLO 实例分割标注文件的路径（已经归一化的多边形）。
    output_file: 输出的 YOLO 目标检测标注文件路径。
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 读取每行标注
            values = list(map(float, line.strip().split()))

            class_id = int(values[0])  # 类别ID
            polygon = values[1:]  # 多边形坐标 (已经归一化)

            # 计算多边形的最小包围框 (Bounding Box)
            x_coords = polygon[0::2]  # 所有的 x 坐标
            y_coords = polygon[1::2]  # 所有的 y 坐标

            x_min = max(min(x_coords), 0)
            x_max = min(max(x_coords), 1)
            y_min = max(min(y_coords), 0)
            y_max = min(max(y_coords), 1)

            # 计算 YOLO 格式下的边界框 (x_center, y_center, width, height)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            x_center = x_min + bbox_width / 2
            y_center = y_min + bbox_height / 2

            # 写入 YOLO 目标检测标注格式: class_id x_center y_center width height
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# endregion


def datasets_sta_yolo(root_dir, save_path):
    data_list = os.listdir(root_dir)

    df = pd.DataFrame(None, columns=['name', 'img_num', 'anno_num', 'task', 'cats'])
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        data_path = os.path.join(data_dir, r'data.yaml')
        with open(data_path) as f:
            data_info = yaml.safe_load(f)
        cats = data_info.get("names", None)

        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        img_num = len(os.listdir(image_dir))
        label_dir = os.path.join(train_dir, 'labels')
        label_num = 0
        label_list = os.listdir(label_dir)
        for label_name in tqdm(label_list):
            label_path = os.path.join(label_dir, label_name)
            if os.path.getsize(label_path) == 0:
                df_label = pd.DataFrame()  # 返回空的 DataFrame
            else:
                df_label = pd.read_csv(label_path, header=None, index_col=None)
            label_num += len(df_label)
        if is_instance_segmentation(label_path):
            df.loc[len(df)] = [data_name, img_num, label_num, 'seg', cats]
        else:
            df.loc[len(df)] = [data_name, img_num, label_num, 'det', cats]
    df.to_csv(save_path)


def dataset_sta_yolo(root_dir, save_name):
    data_list = os.listdir(root_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))

        data_dir = os.path.join(root_dir, data_name)
        data_path = os.path.join(data_dir, save_name)
        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        imgs_list = os.listdir(image_dir)
        df = pd.DataFrame(imgs_list, columns=['name'])
        df['src_name'] = df['name'].str.split('.rf.').str[0]
        print(len(df)-df['src_name'].nunique())

        df.set_index('name', inplace=True)
        df['height'] = None
        df['width'] = None
        for img_name in tqdm(imgs_list):
            img = io.imread(os.path.join(image_dir, img_name))
            df.loc[img_name, 'height'] = img.shape[0]
            df.loc[img_name, 'width'] = img.shape[1]

        sns.jointplot(x='height', y='width', data=df)
        plt.savefig(data_path.replace('.csv', '_shape.png'))
        df['filter'] = False
        df.to_csv(data_path)


def search_aug(root_dir, input_name, output_name):

    data_list = os.listdir(root_dir)

    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        data_path = os.path.join(data_dir, input_name)
        df = pd.read_csv(data_path, header=0, index_col=0)
        df['img_mean'] = 0
        df['aug'] = False
        df_duplicate = df[df['src_name'].duplicated(keep=False)]
        if len(df_duplicate) > 0:
            duplicate_groups = df_duplicate.groupby('src_name')

            for group_name, group in tqdm(duplicate_groups, total=len(duplicate_groups)):
                imgs_name_list = group.index.tolist()

                min_value, min_name = np.inf, None
                for img_name in imgs_name_list:
                    img_path = os.path.join(image_dir, img_name)
                    img = io.imread(img_path)
                    img_mean = img.mean()
                    df.loc[img_name, 'img_mean'] = img_mean
                    df.loc[img_name, 'aug'] = True
                    if img_mean < min_value:
                        min_value = img_mean
                        min_name = img_name
                df.loc[min_name, 'aug'] = False
        df['img_mean'] = df['img_mean'].round(4)
        df['filter'] = df[['aug']].any(axis=1)
        print(np.sum(df['aug']==True), '\n')
        df.to_csv(os.path.join(data_dir, output_name))


def search_shift(root_dir, input_name, output_name):

    data_list = os.listdir(root_dir)

    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        data_path = os.path.join(data_dir, input_name)
        df_src = pd.read_csv(data_path, header=0, index_col=0)
        df_src['swift_next'] = 0
        df_src['swift'] = False

        df = df_src.copy()
        df = df[df['filter'] == False]
        img_current, img_next = None, None
        for idx in tqdm(range(len(df))):
            if idx == len(df)-1:
                break

            img_name_current = df.index[idx]
            if idx == 0:
                img_path_current = os.path.join(image_dir, img_name_current)
                img_current = io.imread(img_path_current)
            else:
                img_current = img_next

            img_name_next = df.index[idx+1]
            img_path_next = os.path.join(image_dir, img_name_next)
            img_next = io.imread(img_path_next)

            shift_p = shift_judge(img_current, img_next)

            df_src.loc[img_name_current, 'swift_next'] = shift_p
        df['swift_next'] = df['swift_next'].round(4)
        df_src['swift'] = df_src['swift_next'] > 0.1
        df_src['filter'] = df_src[['aug', 'swift']].any(axis=1)
        print(np.sum(df_src['swift']==True), '\n')
        df_src.to_csv(os.path.join(data_dir, output_name))


def search_rot(root_dir, input_name, output_name):
    data_list = os.listdir(root_dir)

    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        data_path = os.path.join(data_dir, input_name)
        df_src = pd.read_csv(data_path, header=0, index_col=0)
        df_src['rotate'] = False

        df = df_src.copy()
        df = df[df['filter'] == False]
        img_current, img_next = None, None
        for idx in tqdm(range(len(df))):
            if idx == len(df)-1:
                break

            img_name_current = df.index[idx]
            if idx == 0:
                img_path_current = os.path.join(image_dir, img_name_current)
                img_current = io.imread(img_path_current)
            else:
                img_current = img_next

            img_name_next = df.index[idx+1]
            img_path_next = os.path.join(image_dir, img_name_next)
            img_next = io.imread(img_path_next)

            rot_mir_result = rot_judge(img_current, img_next)

            df_src.loc[img_name_current, 'rotate'] = rot_mir_result

        df_src['filter'] = df_src[['aug', 'swift', 'rotate']].any(axis=1)
        print(np.sum(df_src['rotate']==True), '\n')
        df_src.to_csv(os.path.join(data_dir, output_name))


def search_mirror(root_dir, input_name, output_name):
    data_list = os.listdir(root_dir)

    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        image_dir = os.path.join(train_dir, 'images')
        data_path = os.path.join(data_dir, input_name)
        df_src = pd.read_csv(data_path, header=0, index_col=0)
        df_src['mirror'] = False

        df = df_src.copy()
        df = df[df['filter'] == False]
        img_current, img_next = None, None
        for idx in tqdm(range(len(df))):
            if idx == len(df)-1:
                break

            img_name_current = df.index[idx]
            if idx == 0:
                img_path_current = os.path.join(image_dir, img_name_current)
                img_current = io.imread(img_path_current)
            else:
                img_current = img_next

            img_name_next = df.index[idx+1]
            img_path_next = os.path.join(image_dir, img_name_next)
            img_next = io.imread(img_path_next)

            rot_mir_result = mirror_judge(img_current, img_next)

            df_src.loc[img_name_current, 'mirror'] = rot_mir_result

        df_src['filter'] = df_src[['aug', 'swift', 'rotate', 'mirror']].any(axis=1)
        print(np.sum(df_src['mirror']==True), '\n')
        df_src.to_csv(os.path.join(data_dir, output_name))


def search_intersect_data(root_dir, input_name, output_name, sta_summary_path):
    df = pd.read_csv(sta_summary_path, header=0, index_col=0)
    for idx, row in df.iterrows():
        data_dir = os.path.join(root_dir, row['name'])
        input_csv_path = os.path.join(data_dir, input_name)
        output_csv_path = os.path.join(data_dir, output_name)
        df_data = pd.read_csv(input_csv_path, header=0, index_col=0)
        df_data['intersect'] = False
        df_data.to_csv(output_csv_path)

    for i in range(len(df)):
        print(i, len(df))
        for j in range(i+1, len(df)):
            if df.iloc[j]['cats'] == df.iloc[i]['cats']:
                src_data_dir = os.path.join(root_dir, df.iloc[i]['name'])
                src_imgs_dir = os.path.join(src_data_dir, 'train', 'images')
                src_data_path = os.path.join(src_data_dir, output_name)
                dst_data_dir = os.path.join(root_dir, df.iloc[j]['name'])
                dst_imgs_dir = os.path.join(dst_data_dir, 'train', 'images')
                dst_data_path = os.path.join(dst_data_dir, output_name)
                df_src = pd.read_csv(src_data_path, header=0, index_col=0)
                df_src_keep = df_src[(df_src['aug'] == False) & (df_src['swift'] == False) & (df_src['intersect'] == False)]
                df_dst = pd.read_csv(dst_data_path, header=0, index_col=0)
                df_dst_keep = df_dst[(df_dst['aug'] == False) & (df_dst['swift'] == False) & (df_dst['intersect'] == False)]

                for name_i, row_i in tqdm(df_src_keep.iterrows(), total=len(df_src_keep)):
                    src_name_i = row_i['src_name']
                    df_dst_keep_match = df_dst_keep[df_dst_keep['src_name'] == src_name_i]
                    if len(df_dst_keep_match) > 0:
                        src_img_path = os.path.join(src_imgs_dir, name_i)
                        src_img = io.imread(src_img_path)
                        for name, row in df_dst_keep_match.iterrows():
                            dst_img_path = os.path.join(dst_imgs_dir, name)
                            dst_img = io.imread(dst_img_path)
                            if same_judge(src_img, dst_img):
                                df_dst.loc[name, 'intersect'] = True
                df_dst.to_csv(dst_data_path)

    for idx, row in df.iterrows():
        data_dir = os.path.join(root_dir, row['name'])
        output_csv_path = os.path.join(data_dir, output_name)
        df_data = pd.read_csv(output_csv_path, header=0, index_col=0)
        df_data['filter'] = df_data[['aug', 'swift', 'rotate', 'mirror', 'intersect']].any(axis=1)
        df_data.to_csv(output_csv_path)


def search_color_tinydiffer(root_dir, input_name, output_name, temp_name):
    def compare_images(image1, image2):
        mean_value = euclidean(image1.mean(axis=(0, 1)), image2.mean(axis=(0, 1)))  # Compare mean colors
        # min_value = euclidean(image1.min(axis=(0, 1)), image2.min(axis=(0, 1)))
        # max_value = euclidean(image1.max(axis=(0, 1)), image2.max(axis=(0, 1)))
        return mean_value
    df = pd.read_csv(sta_summary_path, header=0, index_col=0)
    for idx, row in df.iterrows():
        data_dir = os.path.join(root_dir, row['name'])
        input_csv_path = os.path.join(data_dir, input_name)
        output_csv_path = os.path.join(data_dir, output_name)
        df_data = pd.read_csv(input_csv_path, header=0, index_col=0)
        df_data['intersect'] = False
        df_data.to_csv(output_csv_path)

    for i in range(len(df)):
        print(i, len(df))
        src_data_dir = os.path.join(root_dir, df.iloc[i]['name'])
        src_imgs_dir = os.path.join(src_data_dir, 'train', 'images')
        src_data_path = os.path.join(src_data_dir, output_name)
        dst_data_path = os.path.join(src_data_dir, temp_name)
        df_src = pd.read_csv(src_data_path, header=0, index_col=0)
        df_src_keep = df_src[df_src['filter'] == False]
        df_dst = pd.DataFrame(1000, index=df_src_keep.index, columns=df_src_keep.index)

        for i in tqdm(range(len(df_src_keep))):
            src_row = df_src_keep.iloc[i]
            src_name = src_row.name
            src_img_path = os.path.join(src_imgs_dir, src_name)
            src_img = io.imread(src_img_path)
            src_img_lab = rgb2lab(src_img)
            for j in range(i, min(i+20, len(df_src_keep))):
                dst_row = df_src_keep.iloc[j]
                dst_name = dst_row.name
                dst_img_path = os.path.join(src_imgs_dir, dst_name)
                dst_img = io.imread(dst_img_path)
                dst_img_lab = rgb2lab(dst_img)
                color_difference = compare_images(src_img_lab, dst_img_lab)
                df_dst.iloc[i, j] = color_difference
        df_dst.to_csv(dst_data_path)

def remove_data(root_dir, del_name):
    data_list = os.listdir(root_dir)

    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        del_path = os.path.join(data_dir, del_name)
        if not os.path.exists(del_path):
            print(f'{del_name} not in {data_name}!')
        else:
            os.remove(del_path)
            print(f'{del_name} del in {data_name}.')


def get_remained_img(root_dir, input_name, each_show=False):
    all_num, all_num_keep, all_num_drop = 0, 0, 0
    data_list = os.listdir(root_dir)
    for idx, data_name in enumerate(data_list):
        data_dir = os.path.join(root_dir, data_name)
        data_path = os.path.join(data_dir, input_name)

        df_src = pd.read_csv(data_path, header=0, index_col=0)
        data_num = len(df_src)
        all_num += data_num
        data_num_keep = np.sum(df_src['filter']==False)
        all_num_keep += data_num_keep
        data_num_drop = data_num - data_num_keep
        all_num_drop = all_num_drop + data_num_drop
        if each_show:
            print(f"all number - {data_num:6d}, kept number - {data_num_keep:6d}, dropped number - {data_num_drop:6d} of [{data_name}]")

    print(f"all number - {all_num:6d}, kept number - {all_num_keep:6d}, dropped number - {all_num_drop:6d} of [All Dataset]")


def final_copy(src_dir, dst_dir, ref_name):
    data_list = os.listdir(src_dir)
    shutil.rmtree(dst_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        src_data_dir = os.path.join(src_dir, data_name)
        dst_data_dir = os.path.join(dst_dir, data_name)
        src_imgs_dir = os.path.join(src_data_dir, 'train', 'images')
        dst_imgs_dir = os.path.join(dst_data_dir, 'train', 'images')
        src_labels_dir = os.path.join(src_data_dir, 'train', 'labels')
        dst_labels_dir = os.path.join(dst_data_dir, 'train', 'labels')
        os.makedirs(dst_imgs_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)
        src_data_path = os.path.join(src_data_dir, ref_name)
        df = pd.read_csv(src_data_path, header=0, index_col=0)
        df = df[df['filter'] == False]
        for name, row in tqdm(df.iterrows(), total=len(df)):
            src_img_path = os.path.join(src_imgs_dir, name)
            dst_img_path = os.path.join(dst_imgs_dir, name)
            src_label_path = os.path.join(src_labels_dir, Path(name).stem+'.txt')
            dst_label_path = os.path.join(dst_labels_dir, Path(name).stem+'.txt')
            shutil.copyfile(src_img_path, dst_img_path)
            shutil.copyfile(src_label_path, dst_label_path)

def check_det_label(root_dir):
    def remove_sample(error_info, label_path, image_dir):
        base_name = Path(label_path).stem
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        img_remove = False
        for ext in img_extensions:
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                img_remove = True
                break
        if not img_remove:
            IOError(f'cannot find image for {label_path}')

        os.remove(label_path)
        print(f'{error_info} remove {os.path.basename(label_path)}, {os.path.basename(img_path)}')

    data_list = os.listdir(root_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        image_dir = os.path.join(data_dir, 'train', 'images')
        label_dir = os.path.join(data_dir, 'train', 'labels_det')

        label_list = os.listdir(label_dir)
        for label_name in label_list:
            label_path = os.path.join(label_dir, label_name)
            if os.path.getsize(label_path) == 0:
                remove_sample(error_info='(empty)', label_path=label_path, image_dir=image_dir)
            else:
                try:
                    df = pd.read_csv(label_path, header=None, index_col=None, sep=' ')
                    if df.shape[1] != 5:
                        remove_sample(error_info=df.shape, label_path=label_path, image_dir=image_dir)
                except Exception as e:
                    remove_sample(error_info=f'(error : {e})', label_path=label_path, image_dir=image_dir)

def dataset_merge(src_dir, dst_dir, sta_summary_path, categories, based_on_updated_labels=False):
    def copy_sample(src_image_path, dst_image_path, src_label_path, dst_label_path, categories_mapping, category):
        df_label = pd.read_csv(src_label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'h', 'w'], sep=' ')
        df_label['cat_id'] = df_label['cat_id'].apply(lambda x: categories_mapping[category[x]])
        df_label = df_label[~df_label['cat_id'].isin([categories_mapping['background'], categories_mapping['mix']])]
        if len(df_label) > 0:
            df_label.to_csv(dst_label_path, header=False, index=False, sep=' ')
            shutil.copyfile(src_image_path, dst_image_path)

    os.makedirs(dst_dir, exist_ok=True)
    if based_on_updated_labels:
        cat_column = 'cats_update'
        cat_check_column = 'cats_check_update'
        label_dir_name = 'labels_det_update'
    else:
        cat_column = 'cat'
        cat_check_column = 'cats_check'
        label_dir_name = 'labels_det'

    categories_mapping = {category: idx for idx, category in enumerate(categories)}

    shutil.rmtree(dst_dir)
    dst_images_dir = os.path.join(dst_dir, 'images')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    df_sta = pd.read_csv(sta_summary_path, header=0, index_col=0)
    df_sta[cat_column] = df_sta[cat_column].apply(ast.literal_eval)
    df_sta[cat_check_column] = df_sta[cat_check_column].apply(ast.literal_eval)

    # region check
    cats_lengths = df_sta[cat_column].apply(len)
    cats_check_lengths = df_sta[cat_check_column].apply(len)
    mismatch = df_sta[cats_lengths != cats_check_lengths]
    if len(mismatch) > 0:
        print(mismatch)

    def check_categories(cats_list):
        return all(item in categories for item in cats_list)
    invalid_records = df_sta[~df_sta[cat_check_column].apply(check_categories)]
    if len(invalid_records) > 0:
        print(invalid_records)
    # endregion


    for idx, row in df_sta.iterrows():
        category = row[cat_check_column]
        data_name = row['name']
        print(f'{idx}/{len(df_sta)}, {data_name}, category: {category}')

        src_data_dir = os.path.join(src_dir, data_name)
        src_images_dir = os.path.join(src_data_dir, 'train', 'images')
        src_labels_dir = os.path.join(src_data_dir, 'train', label_dir_name)

        img_list = os.listdir(src_images_dir)
        for img_name in tqdm(img_list):
            src_image_path = os.path.join(src_images_dir, img_name)
            src_label_path = os.path.join(src_labels_dir, Path(img_name).stem+'.txt')
            dst_image_path = os.path.join(dst_images_dir, img_name)
            dst_label_path = os.path.join(dst_labels_dir, Path(img_name).stem+'.txt')

            copy_sample(src_image_path, dst_image_path, src_label_path, dst_label_path, categories_mapping, category)

def convert_seg2det(root_dir, ref_path):
    data_list = os.listdir(root_dir)
    df = pd.read_csv(ref_path, header=0, index_col=0)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        task = df['task'][idx]
        data_dir = os.path.join(root_dir, data_name)
        label_dir = os.path.join(data_dir, 'train', 'labels')
        label_seg_dir = os.path.join(data_dir, 'train', 'labels_seg')
        label_det_dir = os.path.join(data_dir, 'train', 'labels_det')
        os.makedirs(label_seg_dir, exist_ok=True)
        os.makedirs(label_det_dir, exist_ok=True)
        if task == 'det':
            for file_name in tqdm(os.listdir(label_dir)):
                src_path = os.path.join(label_dir, file_name)
                det_path = os.path.join(label_det_dir, file_name)
                shutil.copyfile(src_path, det_path)
        elif task == 'seg':
            for file_name in tqdm(os.listdir(label_dir)):
                src_path = os.path.join(label_dir, file_name)
                seg_path = os.path.join(label_seg_dir, file_name)
                det_path = os.path.join(label_det_dir, file_name)
                shutil.copyfile(src_path, seg_path)
                convert_yolo_segmentation_to_detection(src_path, det_path)
        elif task == 'hybrid':
            for file_name in tqdm(os.listdir(label_dir)):
                src_path = os.path.join(label_dir, file_name)
                det_path = os.path.join(label_det_dir, file_name)
                convert_yolo_hybrid_to_detection(src_path, det_path)
        else:
            raise NotImplementedError(f'P{data_name} is {task} type')

def dataset_vis(root_dir, ref_path):
    data_list = os.listdir(root_dir)
    df = pd.read_csv(ref_path, header=0, index_col=0)
    df['cats'] = df['cats'].apply(ast.literal_eval)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        cats = df['cats'][idx]
        df_cats = pd.DataFrame(cats, columns=['cats'])
        data_dir = os.path.join(root_dir, data_name)
        image_dir = os.path.join(data_dir, 'train', 'images')
        label_dir = os.path.join(data_dir, 'train', 'labels_det')
        vis_dir = os.path.join(data_dir, 'train', 'img_vis')
        crop_dir = os.path.join(data_dir, 'train', 'img_crop')
        class_path = os.path.join(data_dir, 'train', 'class.txt')
        df_cats.to_csv(class_path, header=False, index=False)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(crop_dir, exist_ok=True)

        yolo_data_vis(image_dir, label_dir, vis_dir, class_file=class_path, crop_dir=crop_dir)


'''
butler-defect-yevm2 有衣服？

'''

def get_csv_by_cliped_img(label_dir, img_crop_dir, class_path=None, class_list=None):
    if class_path is not None:
        df = pd.read_csv(class_path, header=None, index_col=None, names=['category'])
        cats = df['category'].to_dict()
    elif class_list is not None:
        cats = {i: name for i, name in enumerate(class_list)}
    else:
        ValueError('class_path or class_list must be specified')

    label_list = os.listdir(label_dir)
    label_list = natsorted(label_list)
    df_label_list = []
    for label_name in tqdm(label_list, desc='Loading labels'):
        label_path = os.path.join(label_dir, label_name)
        df_label = pd.read_csv(label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'w', 'h'], sep=' ')
        df_label['file_name'] = label_name
        df_label['object_id'] = df_label.index
        df_label_list.append(df_label)
    if len(df_label_list) > 0:
        df = pd.concat(df_label_list, axis=0)
        df['cat_name'] = df['cat_id'].map(cats)
        df['updated_cat_name'] = df['cat_name']

        for cat_id in range(len(cats)):
            df_cat = df[df['cat_id'] == cat_id]
            df_cat_path = os.path.join(img_crop_dir, cats[cat_id]+'.csv')
            df_cat.to_csv(df_cat_path, index=False)

def dataset_get_csv_by_cliped_img(root_dir, sta_summary_path):
    df_sta = pd.read_csv(sta_summary_path, header=0, index_col=0)
    df_sta['cats'] = df_sta['cats'].apply(ast.literal_eval)
    df_sta['cats_check'] = df_sta['cats_check'].apply(ast.literal_eval)

    for idx, dataset_name in enumerate(df_sta['name']):
        print(idx, dataset_name)
        data_dir = os.path.join(root_dir, dataset_name, 'train')
        label_dir = os.path.join(data_dir, 'labels_det')
        img_crop_dir = os.path.join(data_dir, 'img_crop')
        class_path = os.path.join(data_dir, 'class.txt')
        get_csv_by_cliped_img(label_dir, img_crop_dir, class_path)

def label_update_by_cliped_img(input_dir, output_dir, img_crop_dir, class_path=None, class_list=None):
    if class_path is not None:
        df = pd.read_csv(class_path, header=None, index_col=None, names=['category'])
        cats = df['category'].to_dict()
    elif class_list is not None:
        cats = {i: name for i, name in enumerate(class_list)}
    else:
        ValueError('class_path or class_list must be specified')

    cats2id = {name: id for id, name in cats.items()}

    os.makedirs(output_dir, exist_ok=True)
    for label_name in tqdm(os.listdir(input_dir), desc='Label copy'):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        shutil.copy(input_path, output_path)

    for cat_id, cat_name in cats.items():
        cat_csv_path = os.path.join(img_crop_dir, f'{cat_name}.csv')
        if not os.path.exists(cat_csv_path):
            continue
        df_cat = pd.read_csv(cat_csv_path, header=0, index_col=0)
        df_cat_differ = df_cat[df_cat['updated_cat_name'] != df_cat['cat_name']]

        df_cat_differ['updated_cat_id'] = df_cat_differ['updated_cat_name'].map(cats2id)
        label_differ_list = df_cat_differ['file_name'].unique()
        for label_differ in tqdm(label_differ_list, desc='Label update'):
            df_label_differ = df_cat_differ[df_cat_differ['file_name'] == label_differ]
            label_path = os.path.join(output_dir, label_differ)
            df_label = pd.read_csv(label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'w', 'h'], sep=' ')
            for idx, row in df_label_differ.iterrows():
                object_id = row['object_id']
                updated_cat_name = row['updated_cat_name']
                df_label.loc[object_id, 'cat_id'] = cats2id[updated_cat_name]
            df_label.to_csv(label_path, header=False, index=False, sep=' ')
        print(f'updated {len(label_differ_list)} files, {len(df_cat_differ)} boxes')

def dataset_label_update_by_cliped_img(root_dir, sta_summary_path):
    df_sta = pd.read_csv(sta_summary_path, header=0, index_col=0)
    df_sta['cats'] = df_sta['cats'].apply(ast.literal_eval)
    df_sta['cats_check'] = df_sta['cats_check'].apply(ast.literal_eval)
    df_sta['cats_update'] = df_sta['cats']
    df_sta['cats_check_update'] = df_sta['cats_check']


    for idx, dataset_name in enumerate(df_sta['name']):

        data_dir = os.path.join(root_dir, dataset_name, 'train')
        label_dir = os.path.join(data_dir, 'labels_det')
        label_update_dir = os.path.join(data_dir, 'labels_det_update')
        img_crop_dir = os.path.join(data_dir, 'img_crop')
        class_path = os.path.join(data_dir, 'class.txt')
        class_update_path = os.path.join(data_dir, 'class_update.txt')

        df = pd.read_csv(class_path, header=None, index_col=None, names=['category'])
        cats = df['category'].to_list()
        added_cats = []
        for cat_name in cats:
            cat_path = os.path.join(img_crop_dir, cat_name+'.csv')
            if not os.path.exists(cat_path):
                continue
            df_cat = pd.read_csv(cat_path)
            updated_cat_names = df_cat['updated_cat_name'].unique()
            for updated_cat_name in updated_cat_names:
                if updated_cat_name not in cats and updated_cat_name not in added_cats:
                    added_cats.append(updated_cat_name)
                    if updated_cat_name not in categories:
                        print(f'{updated_cat_name} not in {categories} for {cat_path}')
        final_cats = cats + added_cats

        src_row = df_sta.loc[idx].to_list()
        src_row[-2] = final_cats
        src_row[-1] = src_row[-1] + added_cats
        df_sta.loc[idx] = src_row
        print(f'\n{idx}, {dataset_name}:\n {cats}\n -->\n {final_cats}\n')

        label_update_by_cliped_img(label_dir, label_update_dir, img_crop_dir, class_list=final_cats)

        df_class_update = pd.DataFrame(src_row[-1], columns=['cat_updated'])
        df_class_update.to_csv(class_update_path, header=False, index=False)

    df_sta.to_csv(sta_summary_path)

if __name__ == '__main__':
    pass
    # region coco codes
    # data_dir = r'E:\data\2024_defect\2024_defect_det'
    # save_path = r'E:\data\2024_defect\2024_defect_det\00data_fuse\sta.csv'
    # save_dir = r'E:\data\2024_defect\2024_defect_det_sta'
    # dataset_sta(data_dir, save_path)
    # dataset_vis(data_dir)
    # val_dir = r'E:\data\2024_defect\2024_defect_det\walldefect\train'
    # val_vis_dir = r'E:\data\2024_defect\2024_defect_det\walldefect\val_vis_select'
    # anno_vis(os.path.join(val_dir, '_annotations.coco.json'), img_dir=val_dir, vis_dir=val_vis_dir, cat_ids=[2])
    # get_root_csv_coco(data_dir, save_dir)
    # endregion


    data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo'
    dst_data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo_final'
    merge_data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo_merge'
    sta_summary_path = r'E:\data\2024_defect\2024_defect_pure_yolo_sta\sta_summary.csv'
    # datasets_sta_yolo(data_dir, sta_summary_path)

    # dataset_sta_yolo(data_dir, 'data.csv')
    #
    # search_aug(data_dir, 'data.csv', 'data_rmaug.csv')
    #
    # search_shift(data_dir, 'data_rmaug.csv', 'data_rmaug_rmshift.csv')
    #
    # search_rot(data_dir, 'data_rmaug_rmshift.csv', 'data_rmaug_rmshift_rmrotate.csv')
    #
    # search_mirror(data_dir, 'data_rmaug_rmshift_rmrotate.csv', 'data_rmaug_rmshift_rmrotate_rmmirror.csv')

    # search_intersect_data(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror.csv', 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv', sta_summary_path)

    # search_color_tinydiffer(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv', 'data_rmaug_rmshift_rmrotate_rmmirror_rminter_rmcolor.csv', 'color_difference.csv')


    # get_remained_img(data_dir, 'data.csv')
    # get_remained_img(data_dir, 'data_rmaug.csv')
    # get_remained_img(data_dir, 'data_rmaug_rmshift.csv')
    # get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate.csv')
    # get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror.csv')
    # get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv')

    # remove_data(data_dir, 'data.csv')

    # final_copy(data_dir, dst_data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv')



    # convert_seg2det(dst_data_dir, sta_summary_path)
    # check_det_label(dst_data_dir)

    # dataset_vis(dst_data_dir, sta_summary_path)

    # dataset_get_csv_by_cliped_img(dst_data_dir, sta_summary_path)

    # dataset_label_update_by_cliped_img(dst_data_dir, sta_summary_path)

    dataset_merge(dst_data_dir, merge_data_dir, sta_summary_path, categories, based_on_updated_labels=True)
