import os
import shutil

from tqdm import tqdm
from isds_tool.PS_data.yolo_tools import random_select, split_add
from isds_tool.PS_data.att_tools import get_all_category

def files_name(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        output_name = file_name.split('_')[0]+'1'+file_name.split('_')[1]
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, output_name)
        shutil.copy(input_path, output_path)
if __name__ == '__main__':
    pass
    # root_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v2'
    # input_image_dir = os.path.join(root_dir, 'images')
    # output_image_dir = os.path.join(root_dir, 'images_rename')
    # input_label_dir = os.path.join(root_dir, 'labels')
    # output_label_dir = os.path.join(root_dir, 'labels_rename')
    # files_name(input_image_dir, output_image_dir)
    # files_name(input_label_dir, output_label_dir)

    # data_dir = r'/localnvme/data/bdd/DReality_data/yolo_filter_v2'
    # images_dir = os.path.join(data_dir, 'images')
    # labels_dir = os.path.join(data_dir, 'labels')
    # labels_sta_dir = os.path.join(data_dir, 'labels_sta')
    # class_file = os.path.join(data_dir, 'class.txt')
    random_select(r'/localnvme/data/bdd/HMT0211/rgb_yolo')
    # train_v1_path = r'/localnvme/data/bdd/DReality_data/yolo_filter_v2/train_v2.txt'
    # train_v2_path = r'/localnvme/data/bdd/DReality_data/yolo_filter_v2/train_v3.txt'
    # add_image_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v2/images_rename'
    # split_add(train_v1_path, add_image_dir, train_v2_path)
    #
    # base_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v2'
    # class_file = os.path.join(base_dir, 'class.txt')
    # infer_list = [
    #     'images_infer',
    #     'images_infer2',
    #     'images_infer3',
    #     'images_infer4',
    #     'images_infer5',
    #     'images_infer6',
    # ]
    # for infer_name in infer_list:
    #     infer_dir = os.path.join(base_dir, infer_name, 'labels')
    #     get_all_category(infer_dir, ref_txt=None, classes=class_file, attributes=None, with_conf=False, conf_threshold=0.01,
    #                      filter_small=None)