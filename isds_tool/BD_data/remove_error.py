import os
import shutil
from pathlib import Path

from tqdm import tqdm


def find_error(error_dir):
    file_list = os.listdir(error_dir)
    file_list = [Path(file_name).stem for file_name in file_list]
    return file_list

def remove_file(input_dir, output_dir, ref_list):
    os.makedirs(output_dir, exist_ok=True)
    input_list = os.listdir(input_dir)
    count = 0
    for file_name in tqdm(input_list):
        file_name_stem = Path(file_name).stem
        if file_name_stem in ref_list:
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            shutil.move(input_path, output_path)
            count += 1
    print(f'Removed {count} files in {len(input_list)} dir, according to {len(ref_list)}')

if __name__ == '__main__':
    pass
    error_dir1 = r'E:\data\202502_signboard\data_annotation\partial_label'
    error_dir2 = r'E:\data\202502_signboard\data_annotation\question_sample'
    input_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images_re'
    input_label_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\labels'
    output_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images_re_filter'
    output_label_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\labels_filter'

    error_list1 = find_error(error_dir1)
    error_list2 = find_error(error_dir2)
    error_list = list(set(error_list1 + error_list2))
    remove_file(input_image_dir, output_image_dir, error_list)
    remove_file(input_label_dir, output_label_dir, error_list)