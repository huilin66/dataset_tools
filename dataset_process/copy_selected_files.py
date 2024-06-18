import os
import os.path as osp
import shutil

from tqdm import tqdm

def img_search(input_dir, output_dir, key_str):
    output_dir = osp.join(output_dir, key_str)
    os.makedirs(output_dir, exist_ok=True)

    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        if not key_str in file_name:
            continue
        input_path = osp.join(input_dir, file_name)
        output_path = osp.join(output_dir, file_name)
        shutil.copyfile(input_path, output_path)

if __name__ == '__main__':
    img_search(r'E:\data\0111_testdata\data\img_s',
               r'E:\data\0111_testdata\data\img_search',
               key_str='34F')