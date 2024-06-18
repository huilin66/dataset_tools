"""
2021/1/24
COCO 格式的数据集转化为 YOLO 格式的数据集，源代码采取遍历方式，太慢，
这里改进了一下时间复杂度，从O(nm)改为O(n+m)，但是牺牲了一些内存占用
--json_path 输入的json文件路径
--save_path 保存的文件夹名字，默认为当前目录下的labels。
"""

import os
import json
import shutil

import pandas as pd
from tqdm import tqdm
import argparse
import random
import xml.etree.ElementTree as ET
import numpy as np

def convert_voc2yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(input_folder, image_id, output_folder):
    if not os.path.exists('%s/labels/' % (output_folder)):
        os.makedirs('%s/labels/' % (output_folder))
    in_file = open('%s/Annotations/%s.xml'%(input_folder, image_id))
    out_file = open('%s/labels/%s.txt'%(output_folder, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        if cls_id != 0:
            print(out_file)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_voc2yolo((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



def convert_coco2yolo(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def coco2yolo(json_file, labels_dir, img_dir_src, img_dir_dst, txt_path):
    data = json.load(open(json_file, 'r'))
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    if not os.path.exists(img_dir_dst):
        os.makedirs(img_dir_dst)
    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    for i, category in enumerate(data['categories']):
        id_map[category['id']] = i

    # 通过事先建表来降低时间复杂度
    max_id = 0
    for img in data['images']:
        max_id = max(max_id, img['id'])
    # 注意这里不能写作 [[]]*(max_id+1)，否则列表内的空列表共享地址
    img_ann_dict = [[] for i in range(max_id + 1)]
    for i, ann in enumerate(data['annotations']):
        img_ann_dict[ann['image_id']].append(i)


    df = pd.DataFrame(None, columns=['file_path'])
    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(labels_dir, ana_txt_name), 'w')

        # 这里可以直接查表而无需重复遍历
        for ann_id in img_ann_dict[img_id]:
            ann = data['annotations'][ann_id]
            box = convert_coco2yolo((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        img_path_src, img_path_dst = os.path.join(img_dir_src, filename), os.path.join(img_dir_dst, filename)
        shutil.copy(img_path_src, img_path_dst)
        df.loc[len(df)] = img_path_dst

    df.to_csv(txt_path, header=None, index=None)



def voc2yolo(anno_dir, save_dir, img_dir, train_percent = 0.1):
    save_dir_r = os.path.dirname(save_dir)
    # 训练集和验证集的比例分配
    # trainval_percent = 1-train_percent

    # 生成的txt文件存放路径
    total_xml = os.listdir(anno_dir)

    num = len(total_xml)
    num_list = list(range(num))
    np.random.shuffle(num_list)
    train_len = int(num * train_percent)
    num_train = num_list[:train_len]
    num_val = num_list[train_len:]

    # tv = int(num * (1-train_percent))
    # tr = int(tv * train_percent)
    # trainval = random.sample(list, tv)
    # train = random.sample(trainval, tr)


    # ftrainval = open(os.path.join(save_dir, 'trainval.txt'), 'w')
    # ftest = open(os.path.join(save_dir, 'test.txt'), 'w')
    ftrain = open(os.path.join(save_dir, 'train.txt'), 'w')
    fval = open(os.path.join(save_dir, 'val.txt'), 'w')

    # ftrainval_r = open(os.path.join(save_dir_r, 'trainval.txt'), 'w')
    # ftest_r = open(os.path.join(save_dir_r, 'test.txt'), 'w')
    ftrain_r = open(os.path.join(save_dir_r, 'train.txt'), 'w')
    fval_r = open(os.path.join(save_dir_r, 'val.txt'), 'w')

    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        full_path = os.path.join(img_dir, name.replace('\n', '',) + '.jpg\n')
        # if i in trainval:
        #     ftrainval.write(name)
        #     ftrainval_r.write(full_path)
        #     if i in train:
        #         ftest.write(name)
        #         ftest_r.write(full_path)
        #     else:
        #         fval.write(name)
        #         fval_r.write(full_path)
        if i not in num_train:
            fval.write(name)
            fval_r.write(full_path)
        else:
            ftrain.write(name)
            ftrain_r.write(full_path)

    # ftrainval.close()
    ftrain.close()
    fval.close()
    # ftest.close()

    # ftrainval_r.close()
    ftrain_r.close()
    fval_r.close()
    # ftest_r.close()

if __name__ == '__main__':
    pass
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_select\annotations\instance_train.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\train',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_select\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_select\images\train',
    #           txt_path = r'E:\data\1211_monhkok\mk_merge\yolo_select\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_select\annotations\instance_val.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\val',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_select\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_select\images\val',
    #           txt_path=r'E:\data\1211_monhkok\mk_merge\yolo_select\val.txt',
    #           )
    #
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_select_add\annotations\instance_train.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_select_add\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\all',
    #           txt_path = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_select_add\annotations\instance_val.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_select_add\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\all',
    #           txt_path=r'E:\data\1211_monhkok\mk_merge\yolo_select_add\val.txt',
    #           )
    #
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_add\annotations\instance_train.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_add\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_add\images\all',
    #           txt_path = r'E:\data\1211_monhkok\mk_merge\yolo_add\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco_add\annotations\instance_val.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco_add\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo_add\images\all',
    #           txt_path=r'E:\data\1211_monhkok\mk_merge\yolo_add\val.txt',
    #           )
    #
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco\annotations\instance_train.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo\images\all',
    #           txt_path = r'E:\data\1211_monhkok\mk_merge\yolo\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\1211_monhkok\mk_merge\coco\annotations\instance_val.json',
    #           labels_dir=r'E:\data\1211_monhkok\mk_merge\yolo\labels\all',
    #           img_dir_src=r'E:\data\1211_monhkok\mk_merge\coco\img',
    #           img_dir_dst=r'E:\data\1211_monhkok\mk_merge\yolo\images\all',
    #           txt_path=r'E:\data\1211_monhkok\mk_merge\yolo\val.txt',
    #           )

    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\annotations\instance_train.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\labels\all',
    #           img_dir_src=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\images_train_w',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\images\all',
    #           txt_path = r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\annotations\instance_val.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\labels\all',
    #           img_dir_src=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\images_val_w',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\images\all',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt\val.txt',
    #           )


    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\annotations\instance_train1.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\labels\all',
    #           img_dir_src=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\images_train_w1',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\images\all',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\train.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\annotations\instance_val1.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\labels\all',
    #           img_dir_src=r'E:\data\0111_testdata\data_labeled4254\coco6s640_wt\images_val_w1',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\images\all',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6s640_wt1\val.txt',
    #           )
    # coco2yolo(json_file=r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\valid\_annotations.coco.json',
    #           labels_dir=r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\yolo\labels',
    #           img_dir_src= r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\valid',
    #           img_dir_dst=r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\yolo\images',
    #           txt_path=r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\yolo\val.txt',
    #           )


    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco5_w\annotations\instance_train.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\labels',
    #           img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco5_w\images_train',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\images',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\train.txt',
    #           )
    #
    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco5_w\annotations\instance_val.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\labels',
    #           img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco5_w\images_val',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\images',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo5_w\val.txt',
    #           )


    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s1280_w\annotations\instance_train.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\labels',
    #           img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco6s1280_w\images_train',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\images',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\train.txt',
    #           )
    #
    # coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6s1280_w\annotations\instance_val.json',
    #           labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\labels',
    #           img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco6s1280_w\images_val',
    #           img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\images',
    #           txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6s1280_w\val.txt',
    #           )


    coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6r1280_w\annotations\instance_train.json',
              labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\labels',
              img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco6r1280_w\images_train',
              img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\images',
              txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\train.txt',
              )

    coco2yolo(json_file=r'E:\data\0111_testdata\data_labeled4254\coco6r1280_w\annotations\instance_val.json',
              labels_dir=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\labels',
              img_dir_src= r'E:\data\0111_testdata\data_labeled4254\coco6r1280_w\images_val',
              img_dir_dst=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\images',
              txt_path=r'E:\data\0111_testdata\data_labeled4254\yolo6r1280_w\val.txt',
              )