# -*- coding: utf-8 -*-
# @File             : json_merge.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/8/1 10:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/8/1 10:25
'''
@File Description:
# 合并json文件，可以通过merge_keys控制合并的字段, 默认合并'images', 'annotations'字段
'''

import json
import argparse
import os
import json
import shutil
from tqdm import tqdm


categories_final = [
    {
        "id": 0,
        "name": "background",
        "supercategory": "none"
    },
    {
        "id": 1,
        "name": "crack",
        "supercategory": "Cracks-and-spalling"
    },
    {
        "id": 2,
        "name": "mold",
        "supercategory": "Cracks-and-spalling"
    },
    {
        "id": 3,
        "name": "peeling_paint",
        "supercategory": "Cracks-and-spalling"
    },
    {
        "id": 4,
        "name": "stairstep_crack",
        "supercategory": "Cracks-and-spalling"
    },
    {
        "id": 5,
        "name": "water_seepage",
        "supercategory": "Cracks-and-spalling"
    },
    {
        "id": 6,
        "name": "spall",
        "supercategory": "Cracks-and-spalling"
    }
]
ROOT_PATH = r'E:\Huilin\2308_concretespalling\data'
data_key_list = [
    'BuildingDefectOnWalls', '200im', '400img', 'CracksAndSpalling',
    'oldbuildingdamagedetection', 'defectdetection', 'walldefect',
]
categories_list = [
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, # background, peeling, spalling, hollowing, crack, mold
    {0: 0, 1: 4, 2: 5, 3: 1, 4: 4, 5: 5, 6: 2}, # background, crack, mold, peeling_paint, stairstep_crack, water_seepage, spalling
    # {0: 0, 1: 1, 2: 6},
    # {0: 0, 1: 1, 2: 6, 3: 2},
]
def json_load(js_path):
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)
    return data

def json_save(js_path, data):
    with open(js_path, 'w') as save_f:
        p(data, save_f)


def js_merge(json_list, js_merge_path, img_list, merge_dir, img_gap=10000, anno_gap=10000):
    print('Merge'.center(100, '-'))
    print()


    images_new = []
    annos_new = []
    for idx, train_js in enumerate(json_list):
        data = json_load(train_js)
        categories, images, annotations = data['categories'], data['images'], data['annotations']

        # data_prefix = 'subdata%d_'%idx
        data_prefix = ''
        img_records = []
        for img_record in tqdm(images, desc='img %s' % (data_prefix)):
            img_record['id'] += img_gap*idx
            src_name = img_record['file_name']
            dst_name = data_prefix + src_name
            img_record['file_name'] = dst_name
            img_records.append(img_record)
            src_path = os.path.join(img_list[idx], src_name)
            dst_path = os.path.join(merge_dir, dst_name)
            shutil.copy(src_path, dst_path)
        anno_records = []
        for anno_record in tqdm(annotations, desc='anno %s' % (data_prefix)):
            anno_record['id'] += anno_gap*idx
            anno_record['image_id'] += anno_gap * idx
            anno_record['category_id'] = categories_list[idx][anno_record['category_id']]
            anno_records.append(anno_record)
        images_new += img_records
        annos_new += anno_records

    data_new = {}
    data_new['images'] = images_new
    data_new['annotations'] = annos_new
    data_new['categories'] = categories_final
    json_save(js_merge_path, data_new)


def js_merge_simple(js1_path, js2_path, js_merge_path, merge_keys=['images', 'annotations']):
    print('Merge'.center(100, '-'))
    print()

    print('json read...\n')
    with open(js1_path, 'r') as load_f:
        data1 = json.load(load_f)
    with open(js2_path, 'r') as load_f:
        data2 = json.load(load_f)

    print('json merge...')
    data = {}
    for k, v in data1.items():
        if k not in merge_keys:
            data[k] = v
            print(k)
        else:
            data[k] = data1[k] + data2[k]
            print(k, 'merge!')
    print()

    print('json save...\n')
    data_str = json.dumps(data, ensure_ascii=False)
    with open(js_merge_path, 'w', encoding='utf-8') as save_f:
        save_f.write(data_str)

    print('finish!')

if __name__ == '__main__':
    pass
    # js_t = r'E:\data\ConcreteDefectMerge\train\_annotations.coco.json'
    # js_v = r'E:\data\ConcreteDefectMerge\valid\_annotations.coco.json'
    # js_a = r'E:\data\ConcreteDefectMerge\all\instance_all.json'
    # js_merge_simple(js_t, js_v, js_a)

    # json_list = [
    #     r'E:\data\1211_monhkok\mk_merge\coco\annotations\instance_train.json',
    #     r'E:\data\ConcreteDefectMerge\all\instance_all.json',
    # ]
    # js_merge_path = r'E:\data\1211_monhkok\mk_merge\coco_add\annotations\instance_train.json'
    # img_list = [
    #     r'E:\data\1211_monhkok\mk_merge\coco\img',
    #     r'E:\data\ConcreteDefectMerge\all',
    # ]
    # merge_dir = r'E:\data\1211_monhkok\mk_merge\coco_add\img'
    # js_merge(json_list, js_merge_path, img_list, merge_dir, img_gap=10000, anno_gap=10000)

    json_list = [
        r'E:\data\1211_monhkok\mk_merge\coco_select\annotations\instance_train.json',
        r'E:\data\ConcreteDefectMerge\all\instance_all.json',
    ]
    js_merge_path = r'E:\data\1211_monhkok\mk_merge\coco_select_add\annotations\instance_train.json'
    img_list = [
        r'E:\data\1211_monhkok\mk_merge\coco_select\img',
        r'E:\data\ConcreteDefectMerge\all',
    ]
    merge_dir = r'E:\data\1211_monhkok\mk_merge\coco_select_add\img'
    js_merge(json_list, js_merge_path, img_list, merge_dir, img_gap=10000, anno_gap=10000)