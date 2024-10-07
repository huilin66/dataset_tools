import os

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

def datasets_sta_yolo(root_dir, save_path):
    data_list = os.listdir(root_dir)

    df = pd.DataFrame(None, columns=['name', 'img_num', 'box_num','cats'])
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
        df.loc[len(df)] = [data_name, img_num, label_num, cats]

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
        print(np.sum(df['aug']==True))
        df.to_csv(os.path.join(data_dir, output_name))


def search_shift(root_dir, input_name, output_name):
    def phase_correlation(imageA, imageB):
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
        df = df[df['aug'] != True]
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

            shift_p = phase_correlation(img_current, img_next)

            df_src.loc[img_name_current, 'swift_next'] = shift_p
        df['swift_next'] = df['swift_next'].round(4)
        df_src['swift'] = df_src['swift_next'] > 0.1

        print(np.sum(df_src['swift']==True))
        df_src.to_csv(os.path.join(data_dir, output_name))


def get_remained_img(root_dir, input_name, output_name, remove_columns):
    all_num, all_num_keep, all_num_drop = 0, 0, 0
    data_list = os.listdir(root_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))

        data_dir = os.path.join(root_dir, data_name)
        data_path = os.path.join(data_dir, input_name)

        df_src = pd.read_csv(data_path, header=0, index_col=0)
        df_src['remove'] = df_src[remove_columns].any(axis=1)
        data_num = len(df_src)
        all_num += data_num
        data_num_keep = np.sum(df_src['remove']==False)
        all_num_keep += data_num_keep
        data_num_drop = data_num - data_num_keep
        all_num_drop = all_num_drop + data_num_drop
        print(data_num, data_num_keep, data_num_drop)

    print(all_num, all_num_keep, all_num_drop)


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
    # save_path = r'E:\data\2024_defect\2024_defect_pure_yolo_sta\sta.csv'
    # datasets_sta_yolo(data_dir, save_path)

    # dataset_sta_yolo(data_dir, 'data.csv')

    # search_aug(data_dir, 'data.csv', 'data_rmaug.csv')

    # search_shift(data_dir, 'data_rmaug.csv', 'data_rmaug_rmshift.csv')

    get_remained_img(data_dir, 'data_rmaug_rmshift.csv', 'data_rmaug_rmshift.csv', ['aug', 'swift'])