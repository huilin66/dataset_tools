import os
import shutil

from tqdm import tqdm
from pathlib import Path

def get_ref_list(input_dir):
    ref_list = [Path(file_name).stem for file_name in os.listdir(input_dir)]
    return ref_list


def copy_by_ref(input_dir, output_dir, ref_list):
    input_list = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for input_name in tqdm(input_list):
        input_name_stem = Path(input_name).stem
        if input_name_stem in ref_list:
            input_path = os.path.join(input_dir, input_name)
            output_path = os.path.join(output_dir, input_name)
            shutil.copy(input_path, output_path)

def assign(input_dir, output_dir1, output_dir2):
    file_list = os.listdir(input_dir)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    for i in range(0, len(file_list), 2):
        file_name1 = file_list[i]
        file_name2 = file_list[i + 1]
        input_path1 = os.path.join(input_dir, file_name1)
        input_path2 = os.path.join(input_dir, file_name2)
        output_path1 = os.path.join(output_dir1, file_name1)
        output_path2 = os.path.join(output_dir2, file_name2)
        shutil.copyfile(input_path1, output_path1)
        shutil.copyfile(input_path2, output_path2)


if __name__ == '__main__':
    pass
    # ref_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\question_sample'
    # input_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images'
    # input_json_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\.json'
    # output_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images'
    # output_json_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\.json'
    # ref_list = get_ref_list(ref_dir)
    # copy_by_ref(input_image_dir, output_image_dir, ref_list)
    # copy_by_ref(input_json_dir, output_json_dir, ref_list)

    # ref_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\partial_label'
    # input_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images'
    # input_json_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\.json'
    # output_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images'
    # output_json_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\.json'
    # ref_list = get_ref_list(ref_dir)
    # copy_by_ref(input_image_dir, output_image_dir, ref_list)
    # copy_by_ref(input_json_dir, output_json_dir, ref_list)


    input_image_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images'
    output_image1_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images1'
    output_image2_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images2'
    input_json_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\images'
    output_json1_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\json1'
    output_json2_dir = r'E:\data\202502_signboard\data_annotation\annotation_error\json2'
    assign(input_image_dir, output_image1_dir, output_image2_dir)
    assign(input_json_dir, output_json1_dir, output_json2_dir)