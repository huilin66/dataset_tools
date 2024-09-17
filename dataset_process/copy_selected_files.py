import os
import os.path as osp
import shutil

from tqdm import tqdm

keep_nums1 = [
    768, 860, 1050, 1072, 1152, 1312, 1430, 1672,
    # 2038,
]

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


def imgs_search(input_dir, output_dir, key_strs):
    os.makedirs(output_dir, exist_ok=True)

    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        if not file_name in key_strs:
            continue
        input_path = osp.join(input_dir, file_name)
        output_path = osp.join(output_dir, file_name)
        shutil.copyfile(input_path, output_path)

def imgs_copy(input_dirs, output_dirs, key_strs):
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        os.makedirs(output_dir, exist_ok=True)
        for key_str in key_strs:
            input_path = osp.join(input_dir, key_str)
            output_path = osp.join(output_dir, key_str)
            shutil.copyfile(input_path, output_path)

if __name__ == '__main__':
    pass
    # img_search(r'E:\data\0111_testdata\data\img_s',
    #            r'E:\data\0111_testdata\data\img_search',
    #            key_str='34F')

    # FLIR0602.png
    # key_strs = ['FLIR%04d.png'%num for num in keep_nums1]
    # imgs_search(
    #     input_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\cat_show',
    #     output_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\select_show3',
    #     key_strs=key_strs,
    # )
    key_strs = ['FLIR%04d.png' % num for num in keep_nums1]
    imgs_copy(
        input_dirs=[
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images_vis',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\yolo8',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\yolo9',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\yolo10',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\mayolo'
        ],
        output_dirs=[
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\images',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\images_vis',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo8',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo9',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo10',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\mayolo'
        ],
        key_strs=key_strs
    )