import os
import json
import numpy as np
from skimage import io
from tqdm import tqdm

def img_cat(img_list, dst_path):
    imgs = [io.imread(img_path) for img_path in img_list]
    img = np.concatenate(imgs, axis=-1)
    print(img.shape)
    io.imsave(dst_path, img)

def imgs_cat(dir_list, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    file_list = os.listdir(dir_list[0])
    for file_name in tqdm(file_list):
        img_list = [os.path.join(img_dir, file_name) for img_dir in dir_list]
        dst_path = os.path.join(dst_dir, file_name.replace('.jpg','.tif'))
        img_cat(img_list, dst_path)

def json_img_replace(json_path, dst_path):
    with open(json_path, 'r') as load_f:
        data = json.load(load_f)

    images = data['images']
    for img_record in images:
        img_record['file_name'] = img_record['file_name'].replace('.jpg', '.tif')
    data['images'] = images

    str_json = json.dumps(data, ensure_ascii=False)
    with open(dst_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(str_json)

if __name__ == '__main__':
    pass
    # dir_list = [
    #     r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_val_w',
    #     r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_val_t'
    # ]
    # dst_dir = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_val_wt'
    # dir_list = [
    #     r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_train_w',
    #     r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_train_t'
    # ]
    # dst_dir = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\images_train_wt'
    # imgs_cat(dir_list, dst_dir)

    # json_path = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\annotations\instance_train.json'
    # dst_path = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\annotations\instance_train_wt.json'
    json_path = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\annotations\instance_val.json'
    dst_path = r'E:\data\0111_testdata\data_labeled4254\coco5_wt\annotations\instance_val_wt.json'
    json_img_replace(json_path, dst_path)