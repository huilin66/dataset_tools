import os
import cv2
import json
import numpy as np


# r'1-s2-0-S0950061816309527-gr10_jpg.rf.9f02989b0724eefdea771bbc61fcaeff.jpg'
def count_black_pixels(image_path):
    img = cv2.imread(image_path)
    # 计算黑色像素的数量（RGB值都为0的像素）
    black_pixels = np.sum(np.all(img == [0, 0, 0], axis=-1))
    return black_pixels

def group_select(img_list):
    max_num = 0
    max_path = None
    for img_path in img_list:
        bp = count_black_pixels(img_path)
        if bp>max_num:
            max_num=bp
            max_path = img_path
    return max_path

def name_group(file_list):
    file_groups = {}
    for file_name in file_list:
        strs = file_name.split('.rf.')
        if strs[0] in list(file_groups.keys()):
            file_groups[strs[0]].append(file_name)
        else:
            file_groups[strs[0]] = [file_name]
    return file_groups

def delete_image_and_annotation(coco_json_path, image_folder_path, image_id_to_delete):
    # 加载COCO JSON文件
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # 删除指定的图片元数据
    data['images'] = [img for img in data['images'] if img['id'] != image_id_to_delete]

    # 删除指定图片的所有标注
    data['annotations'] = [anno for anno in data['annotations'] if anno['image_id'] != image_id_to_delete]

    # 将更新后的数据写回JSON文件
    with open(coco_json_path, 'w') as f:
        json.dump(data, f)

    # 删除图片文件
    for filename in os.listdir(image_folder_path):
        if filename.startswith(str(image_id_to_delete)):
            os.remove(os.path.join(image_folder_path, filename))

def rm_aug(input_dir, output_dir):
    file_list = [filename for filename in os.listdir(input_dir) if not filename.endswith('.json')]
    file_groups = name_group(file_list)
    # print(file_groups)
    del_list = []
    for k,v in file_groups:
        img_list = [os.path.join(input_dir, filename) for filename in v]
        src_file = group_select(img_list)
    print('%d -> %d'%(len(file_list), len(file_groups)))


if __name__ == '__main__':
    pass
    input_dir = r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\train'
    output_dir = r'E:\data\2024_defect\2024_defect_det\ConcreteCracksDetection\train_select'
    rm_aug(input_dir, output_dir)