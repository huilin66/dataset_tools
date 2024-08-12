import os
import os.path as osp
import shutil
import pandas as pd
from tqdm import tqdm

def seg_filter(input_dir, copy_dir):
    '''
    remove instance segmentation label from all records
    :param input_dir:
    :param copy_dir:
    :return:
    '''
    os.makedirs(copy_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)
        copy_file = osp.join(copy_dir, file_name)
        # shutil.copy(input_file, copy_file)

        with open(input_file, 'r') as file:
            lines = file.readlines()
        filtered_lines = []
        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            if numbers[1] > 1:
                filtered_lines.append(line)
        with open(copy_file, 'w') as file:
            file.writelines(filtered_lines)

def attribute_remove(input_dir, copy_dir):
    '''
    remove attributes from each record, get detection result
    :param input_dir:
    :param copy_dir:
    :return:
    '''
    os.makedirs(copy_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)
        copy_file = osp.join(copy_dir, file_name)
        # shutil.copy(input_file, copy_file)

        with open(input_file, 'r') as file:
            lines = file.readlines()
        filtered_lines = []
        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            numbers = numbers[:1]+numbers[-4:]
            numbers_str = [str(num) for num in numbers]
            new_line = ' '.join(numbers_str)+'\n'
            filtered_lines.append(new_line)
        with open(copy_file, 'w') as file:
            file.writelines(filtered_lines)

def seg_remove(input_dir, copy_dir):
    '''

    :param input_dir:
    :param copy_dir:
    :return:
    '''
    os.makedirs(copy_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)
        copy_file = osp.join(copy_dir, file_name)
        # shutil.copy(input_file, copy_file)

        with open(input_file, 'r') as file:
            lines = file.readlines()
        filtered_lines = []
        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            numbers = numbers[:1]+numbers[-4:]
            numbers_str = [str(num) for num in numbers]
            new_line = ' '.join(numbers_str)+'\n'
            filtered_lines.append(new_line)
        with open(input_file, 'w') as file:
            file.writelines(filtered_lines)


def att_negative_remove(input_dir, output_dir, att_len=14):
    input_img_dir = osp.join(osp.dirname(input_dir), 'images')
    output_img_dir = osp.join(osp.dirname(output_dir), 'images')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    input_list = os.listdir(input_dir)
    for input_name in tqdm(input_list):
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, input_name)
        input_img_path = osp.join(input_img_dir, input_name.replace('.txt', '.png'))
        output_img_path = osp.join(output_img_dir, input_name.replace('.txt', '.png'))

        df = pd.read_csv(input_path, header=None, index_col=None, sep=' ')

        if att_len==14:
            selected_nums = list(range(1, 2+att_len))
            selected_nums.remove(5)
        else:
            selected_nums = list(range(1, 2+att_len))
        selected_columns = df[selected_nums]
        df_det = df.drop(columns=+selected_columns)

        df_det.to_csv(output_path, header=None, index=None, sep=' ')
        shutil.copy(input_img_path, output_img_path)

if __name__ == '__main__':
    pass
    # seg_filter(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels_all')
    # attribute_remove(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels_att')


    # seg_filter(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels')

    # attribute_remove(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6_det\labels')


    # input_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5\labels'
    # output_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_f\labels'
    # att_negative_remove(input_dir, output_dir)

    # input_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\labels'
    # output_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10f\labels'
    # att_negative_remove(input_dir, output_dir, att_len=10)

    att_negative_remove(
        input_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\labels',
        output_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_det\labels',
        att_len=10
    )