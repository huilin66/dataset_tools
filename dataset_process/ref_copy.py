import os
import shutil
from tqdm import tqdm


def dir_ref_remove(input_dir, output_dir, ref_dir):
    file_list = os.listdir(ref_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.move(input_path, output_path)

def dir_remove(input_dir, output_dir):
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.move(input_path, output_path)

def dir_ref_copy(input_dir, output_dir, ref_dir):
    file_list = os.listdir(ref_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.copy(input_path, output_path)


if __name__ == '__main__':
    pass
    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\images\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\images\train'
    # dir_remove(input_dir, output_dir)
    #
    #
    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select_add\labels\train'
    # dir_remove(input_dir, output_dir)


    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\images\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\images\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\images\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\images\train'
    # dir_remove(input_dir, output_dir)
    #
    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo_add\labels\train'
    # dir_remove(input_dir, output_dir)

    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo\images\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo\images\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\images\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo\images\train'
    # dir_remove(input_dir, output_dir)
    #
    # input_dir = r'E:\data\1211_monhkok\mk_merge\yolo\labels\all'
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo\labels\val'
    # ref_dir = r'E:\data\1211_monhkok\mk_merge\yolo_select\labels\val'
    # dir_ref_remove(input_dir, output_dir, ref_dir)
    # output_dir = r'E:\data\1211_monhkok\mk_merge\yolo\labels\train'
    # dir_remove(input_dir, output_dir)