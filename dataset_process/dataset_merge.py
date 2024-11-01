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

from data_vis.yolo_vis import yolo_data_vis
# region coco datasets
ROOT_PATH = r'E:\data\2024_defect\2024_defect_det'

error_set = {
    'ConcreteCracksDetection':['data repeat'], # with hole category
    'Dam_data.v14i.coco': ['hole category mixed with peeling and spalling'], # with hole category, mixed with peeling and spalling
    'defectdetection': ['low resolution', 'Contamination is confusing'],
    'Defects.v7-last-one.coco': ['cropped data'],
    'detr_crack_dataset.v1i.coco': ['too much crack'],
    'dsa.v1i.coco': ['blister mixed with water seepage'],
    'new dataset.v3i.coco': ['too fragment'],
    'tile.v6i.coco': ['too fragment'],
    'wall.v1i.coco': ['few data'],
    # 'walldefect': ['peeling is  crack'] # useful crack
}

categories_map = {
    '200im': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
    '400img': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
    'Building Defect.v3i.coco': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
}
categories_final = [
    {
        "id": 0,
        "name": "background",
        "supercategory": "none"
    },
    {
        "id": 1,
        "name": "crack",
        "supercategory": "defect"
    },
    {
        "id": 2,
        "name": "concrete_spalling",
        "supercategory": "defect"
    },
    {
        "id": 3,
        "name": "finishes_peeling",
        "supercategory": "defect"
    },
    {
        "id": 4,
        "name": "water_seepage",
        "supercategory": "defect"
    },
    {
        "id": 5,
        "name": "stain",
        "supercategory": "defect"
    },
    {
        "id": 6,
        "name": "vegetation",
        "supercategory": "defect"
    }
]

# endregion

# region coco tools

def json_load(js_path):
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)
    return data

def json_save(js_path, data):
    with open(js_path, 'w') as save_f:
        json.dump(data, save_f)


def dataset_merge_coco(dst_dir, data_prefixs=None, merge_dir='train', img_gap=10000, anno_gap=10000):
    dst_dir = os.path.join(dst_dir, merge_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(os.path.join(dst_dir, merge_dir))
    dst_json = os.path.join(dst_dir, '_annotations.coco.json')

    data_list = os.listdir(dst_dir)
    data_list.remove('00data_fuse')
    data_key_list = [i for i in data_list if i not in b]


    # data_key_list = [
    #     '200im', '400img', 'Building Defect.v3i.coco',
    #     'oldbuildingdamagedetection', 'defectdetection', 'walldefect',
    #     # '200im', 'defectdetection',
    # ]
    # img_list = [
    #     os.path.join(ROOT_PATH, data_key, merge_dir) for data_key in data_key_list
    # ]
    # json_list = [
    #     os.path.join(ROOT_PATH, data_key, merge_dir, '_annotations.coco.json') for data_key in data_key_list
    # ]
    # categories_list = [
    #     {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6, 3: 2},
    #     {0: 0, 1: 1, 2: 2, 3: 3},
    #     # {0: 0, 1: 1, 2: 6},
    #     # {0: 0, 1: 1, 2: 6, 3: 2},
    # ]
    if data_prefixs is None or len(data_prefixs) != len(data_key_list):
        data_prefixs = ['data%02d_' % idx for idx in range(len(data_key_list))]

    images_new = []
    annos_new = []
    for idx, train_js in enumerate(json_list):
        data = json_load(train_js)
        categories, images, annotations = data['categories'], data['images'], data['annotations']

        data_prefix = data_prefixs[idx]
        img_records = []
        for img_record in tqdm(images, desc='%s img %s' % (merge_dir, data_prefix)):
            img_record['id'] += img_gap*idx
            src_name = img_record['file_name']
            dst_name = data_prefix + src_name
            img_record['file_name'] = dst_name
            img_records.append(img_record)
            src_path = os.path.join(img_list[idx], src_name)
            dst_path = os.path.join(dst_dir, dst_name)
            shutil.copy(src_path, dst_path)
        anno_records = []
        for anno_record in tqdm(annotations, desc='%s anno %s' % (merge_dir, data_prefix)):
            anno_record['id'] += anno_gap*idx
            anno_record['image_id'] += anno_gap * idx
            anno_record['category_id'] =  categories_list[idx][anno_record['category_id']]
            anno_records.append(anno_record)
        images_new += img_records
        annos_new += anno_records
    data_new = {}
    data_new['images'] = images_new
    data_new['annotations'] = annos_new
    data_new['categories'] = categories_final
    json_save(dst_json, data_new)


def dataset_sta_coco(root_dir, save_path):
    pass

    df = pd.DataFrame(None, columns=['name', 'train_img', 'train_box', 'val_img', 'val_num', 'test_img','test_num', 'cats'])

    dir_list = os.listdir(root_dir)
    dir_list.remove('00data_fuse')
    for data_name in dir_list:
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'valid')
        test_dir = os.path.join(data_dir, 'test')
        train_data = json_load(os.path.join(train_dir, '_annotations.coco.json'))
        val_data = json_load(os.path.join(val_dir, '_annotations.coco.json'))
        test_data = json_load(os.path.join(test_dir, '_annotations.coco.json'))
        cat_list = train_data['categories']
        cats = []
        for category in cat_list:
            cat_name = category['name']
            cats.append(cat_name)
        train_img = len(train_data['images'])
        train_box = len(train_data['annotations'])
        val_img = len(val_data['images'])
        val_box = len(val_data['annotations'])
        test_img = len(test_data['images'])
        test_box = len(test_data['annotations'])
        record = [data_name, train_img, train_box, val_img, val_box, test_img, test_box, cats]
        df.loc[len(df)] = record
    print(df)
    df.to_csv(save_path)


def dataset_vis_coco(root_dir):
    dir_list = os.listdir(root_dir)
    dir_list.remove('00data_fuse')
    for data_name in dir_list:
        data_dir = os.path.join(root_dir, data_name)
        # val_dir = os.path.join(data_dir, 'valid')
        # val_vis_dir = os.path.join(data_dir, 'val_vis')
        # anno_vis(os.path.join(val_dir, '_annotations.coco.json'), img_dir=val_dir, vis_dir=val_vis_dir)
        train_dir = os.path.join(data_dir, 'train')
        train_vis_dir = os.path.join(data_dir, 'train_vis')
        anno_vis(os.path.join(train_dir, '_annotations.coco.json'), img_dir=train_dir, vis_dir=train_vis_dir)


def get_root_csv_coco(data_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dir_list = os.listdir(data_root)
    for data_name in dir_list:
        data_dir = os.path.join(data_root, data_name)
        if os.path.isdir(data_dir):
            data_path = os.path.join(data_dir, r'README.dataset.txt')
            if os.path.exists(data_path):
                save_path = os.path.join(save_dir, data_name+'_url.txt')
                shutil.copy(data_path, save_path)
                print('save to',save_path)
            else:
                print(data_name, 'not exist')

# endregion

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
        if idx<16:
            continue
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



    get_remained_img(data_dir, 'data.csv')
    get_remained_img(data_dir, 'data_rmaug.csv')
    get_remained_img(data_dir, 'data_rmaug_rmshift.csv')
    get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate.csv')
    get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror.csv')
    get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv')

    # remove_data(data_dir, 'data.csv')

    # final_copy(data_dir, dst_data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv')

    # convert_seg2det(dst_data_dir, sta_summary_path)

    # dataset_vis(dst_data_dir, sta_summary_path)
