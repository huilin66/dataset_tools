import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

class_list = [0, 1]

def find_image_with_class(labels_dir, class_list, seg=False, with_att=False):
    label_list = os.listdir(labels_dir)
    for label_name in tqdm(label_list):
        label_path = osp.join(labels_dir, label_name)
        if not seg:
            pass
            # df = pd.read_csv(label_path, names=['class', 'xmin', 'ymin', 'xmax', 'ymax'])
        else:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for id_line, line in enumerate(lines):
                    parts = line.strip().split(' ')
                    class_id = int(parts[0])
                    if class_id in class_list:
                        print(f'find class{class_id} in {label_name}, object {id_line}')


def yolo_classmerge(input_dir):
    '''

    :param input_dir:
    :param copy_dir:
    :return:
    '''

    file_list = os.listdir(input_dir)

    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)


        with open(input_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            if numbers[0] in class_list:
                continue
            else:
                print(file_name, numbers)

def check_mdet(label_dir, len=10):
    label_list = os.listdir(label_dir)
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        print(label_name)
        df = pd.read_csv(label_path, header=None, index_col=None, sep=' ')
        print(df.shape)


def mdet_predict_check(input_dir, ref_dir):
    file_list = os.listdir(ref_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        if not os.path.exists(input_path):
            with open(input_path, 'r') as file:
                pass
        else:
            if os.stat(input_path).st_size == 0:
                pass
            else:
                df = pd.read_csv(input_path, header=None, index_col=None, sep=' ')
                df['drop'] = False
                for idx, row in df.iterrows():
                    if row[df.shape[1]-2] == 0 or row[df.shape[1]-3] == 0:
                        df.at[idx, 'drop'] = True
                df_new = df[df['drop'] == False]
                df_new = df_new.drop(columns=['drop'])
                if len(df_new) != len(df):
                    df_new.to_csv(input_path, header=False, index=False, sep=' ')
                    print(f'{file_name} change from {len(df)} -> {len(df_new)}')


def info_merge(input_dir, save_path):
    label_list = os.listdir(input_dir)
    df = pd.DataFrame(None, columns=['file_name', 'object_id', 'class_id', 'att_num', 'att', 'segment'])
    for label_name in tqdm(label_list):
        label_path = os.path.join(input_dir, label_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                nums = line.strip().split()
                df.loc[len(df)] = [label_name, idx, int(nums[0]), int(nums[1]), nums[2:2+int(nums[1])], nums[2+int(nums[1]):]]
    df.to_csv(save_path, encoding='utf-8-sig')




if __name__ == '__main__':
    pass
    # yolo_classmerge(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6\labels')
    # check_mdet(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels')

    # mdet_predict_check(
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\mayolo_infer',
    #     r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels_val',
    # )
    # info_merge(r'E:\data\202502_signboard\annotation_result_merge\labels_update',
    #            r'E:\data\202502_signboard\annotation_result_merge\info_update.csv')
    find_image_with_class(r'E:\data\202502_signboard\data_annotation\task\task0519_anno\yolo_dataset\labels',
                          class_list=[1], seg=True)