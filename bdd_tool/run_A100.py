import os
# from isds_tool.PS_data.yolo_tools import random_select
from data_vis.yolo_sta import yolo_sta
if __name__ == '__main__':
    data_dir = r'/data/huilin/data/BDD/BD1012'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    labels_sta_dir = os.path.join(data_dir, 'labels_sta')
    class_file = os.path.join(data_dir, 'class.txt')
    # random_select(data_dir)
    yolo_sta(labels_dir, labels_sta_dir, class_file, img_dir=images_dir)