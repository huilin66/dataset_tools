import os
import shutil
from tqdm import tqdm
import sys
sys.path.append(r'E:\repository\dataset_tools')
from dataset_process.deduplication_demo import filter_deduplication


def merge_drone_data_mannual(input_dir, output_dir):
    output_dir_v = os.path.join(output_dir, 'visiable')
    output_dir_t = os.path.join(output_dir, 'thermal')
    os.makedirs(output_dir_v, exist_ok=True)
    os.makedirs(output_dir_t, exist_ok=True)
    sub_dir = os.listdir(input_dir)
    for sub_name in sub_dir:
        sub_dir_path = os.path.join(input_dir, sub_name)
        if not os.path.isdir(sub_dir_path) or not sub_name.startswith('DJI'):
            continue
        file_list = os.listdir(sub_dir_path)
        for file_name in tqdm(file_list, desc=f'Copying files from {sub_name}'):
            file_path = os.path.join(sub_dir_path, file_name)
            if file_path.lower().endswith('_v.jpg'):
                output_path = os.path.join(output_dir_v, file_name)
            elif file_path.lower().endswith('_t.jpg'):
                output_path = os.path.join(output_dir_t, file_name)
            else:
                print(f'Unknown file type: {file_path}')
                continue
            shutil.copy(file_path, output_path)



def select_data_by_gap(input_dir, select_dir, gap_num=5):
    os.makedirs(select_dir, exist_ok=True)

    file_list = os.listdir(input_dir)
    file_list.sort()
    selected_files = []
    for i in range(0, len(file_list), gap_num):
        selected_files.append(file_list[i])
    for file_name in tqdm(selected_files, desc="Selecting files"):
        file_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(select_dir, file_name)
        shutil.copy(file_path, output_path)



if __name__ == '__main__':
    HMT_data_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\collected data'
    data_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data'
    # merge_drone_data_mannual(HMT_data_dir, data_dir)
    vis_dir = os.path.join(data_dir, 'visible')
    gap_num = 3
    # select_data_by_gap(vis_dir, vis_dir+f'_selected_{gap_num}', gap_num)
    # filter_deduplication(vis_dir+f'_selected_{gap_num}', vis_dir+f'_selected_{gap_num}_filter', window_size=10, threshold=0.3)

    gap_num = 4
    HMT_data_dir_t = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\thermal'
    select_data_by_gap(HMT_data_dir_t, HMT_data_dir_t+f'_selected_{gap_num}', gap_num)