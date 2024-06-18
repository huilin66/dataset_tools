import os
import os.path as osp
import shutil

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

if __name__ == '__main__':
    pass
    # seg_filter(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels_all')
    # attribute_remove(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels',
    #            copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\labels_att')


    seg_filter(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels',
               copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels')