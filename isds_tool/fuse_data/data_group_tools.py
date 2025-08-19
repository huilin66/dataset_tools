import os
import json
import shutil

from tqdm import tqdm
import pandas as pd
from pathlib import Path

def data_merge(input_dir, images_dir, labels_dir, json_dir):
    count_list = [0]
    os.makedirs(json_dir, exist_ok=True)
    # os.makedirs(images_dir, exist_ok=True)
    # os.makedirs(labels_dir, exist_ok=True)
    sub_list = os.listdir(input_dir)
    for sub_name in sub_list:
        sub_path = os.path.join(input_dir, sub_name)
        # input_image_dir = os.path.join(sub_path, 'images')
        # input_labels_dir = os.path.join(sub_path, 'labels')
        input_json_dir = os.path.join(sub_path, 'json')
        json_list = os.listdir(input_json_dir)
        for json_name in tqdm(json_list):
            input_json_path = os.path.join(input_json_dir, json_name)
            json_path = os.path.join(json_dir, json_name)
            if 'left' in sub_name:
                json_path = os.path.join(json_dir, Path(json_name).stem + '_left.json')
            if 'right' in sub_name:
                json_path = os.path.join(json_dir, Path(json_name).stem + '_right.json')
            shutil.copy(input_json_path, json_path)
            # input_label_path = os.path.join(input_labels_dir, Path(json_name).stem+'.txt')
            # label_path = os.path.join(labels_dir, Path(json_name).stem+'.txt')
            # shutil.copy(input_label_path, label_path)
            count_list.append(os.path.basename(json_path))
    print(len(set(count_list)))

def check_group_id(input_dir):
    pass
    df = pd.DataFrame(None, columns=['file_name', 'group_id'])
    json_file_list = os.listdir(input_dir)
    for json_name in tqdm(json_file_list):
        json_path = os.path.join(input_dir, json_name)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data['shapes']
        group_id = 0
        for shape in shapes:
            if 'group_id' in shape and shape['group_id'] is not None:
                group_id = max(int(shape['group_id']), group_id)
        df.loc[len(df)] = [json_name, group_id]
    df.to_csv(input_dir+'.csv')

if __name__ == '__main__':
    pass
    root_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\.json'
    # check_group_id(root_dir)

    anno_dir = r'E:\data\202502_signboard\data_annotation\task_result_merge\annos'
    images_dir = r'E:\data\202502_signboard\data_annotation\task_result_merge\images'
    labels_dir = r'E:\data\202502_signboard\data_annotation\task_result_merge\labels'
    json_dir = r'E:\data\202502_signboard\data_annotation\task_result_merge\json'
    data_merge(anno_dir, images_dir, labels_dir, json_dir)
    # check_group_id(json_dir)
