import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from data_vis.yolo_vis import yolo_mdet_vis
from pathlib import Path
from yolo_mask_crop import myolo_crop
categories = ['wall frame', 'wall display', 'projecting frame', 'projecting display', 'hanging frame', 'hanging display', 'other']
attributes = ['deformation', 'broken', 'abandonment', 'corrosion']


# region prepare prediction
def get_ref_list(csv_path):
    df = pd.read_csv(csv_path, header=None, index_col=None, names=['path'])
    file_path_list = df['path'].to_list()
    file_name_list = [Path(os.path.basename(file_path)).stem for file_path in file_path_list]
    return file_name_list


def get_gt_info_mdet(gt_path):
    gt_infos = []
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            img_name = Path(gt_path).stem + '_%d'%idx + '.png'
            gt_info = {}
            gt_info['img_name'] = img_name
            gt_info['category'] = categories[int(parts[0])]
            for i, att in enumerate(attributes):
                gt_info[att] = parts[i+2]
            gt_infos.append(gt_info)
    return gt_infos


def get_img_info_mdet(gt_path, img_dir, description=1):
    def get_defect_name(attributess):
        attributess_list = ["'%s risk'"%attribute_name for attribute_name in attributess]
        names_str = ', '.join(attributess_list)
        return names_str
    def get_defect_info(gt_info, attributes, attributes_names):
        attributess_list = []
        for idx, attribute in enumerate(attributes):
            if bool(int(gt_info[attribute])):
                attributess_list.append("'%s'"%attributes[idx])
        if len(attributess_list) == 0:
            return "no"
        else:
            return ' '.join(attributess_list)
    gt_infos = get_gt_info_mdet(gt_path)
    img_infos = []
    for gt_info in gt_infos:
        img_name = gt_info['img_name']
        img_info = {}
        img_info['id'] = img_name.replace('.jpg', '')
        img_info['image'] = os.path.basename(img_dir) + '/'+ img_name
        # 原始描述
        if description == 1:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nplease describe this image in table format."
                },
                {
                    "from": "gpt",
                    "value": "| property | value |\n" +
                             "| --- | --- |\n" +
                             ''.join(["| %s | %s |\n" % (attribute, bool(int(gt_info[attribute]))) for idx,attribute in enumerate(attributes)])

                },
            ]
        # 修改defect property，使其为正常单词
        elif description == 2:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nplease describe this image in table format."
                },
                {
                    "from": "gpt",
                    "value": "| property | value |\n"
                             "| --- | --- |\n"
                             "| category | %s |\n"%(gt_info['category']) +
                             ''.join(["| %s | %s |\n" % (attribute.replace('_', ' '), bool(int(gt_info[attribute]))) for attribute in attributes])

                },
            ]
        # 修改输入描述与输出描述，使gpt可以理解要描述的是signboard defect， 输出结果也进行相应修改
        elif description == 3:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease describe the defect of the signboard in this image in table format."
                },
                {
                    "from": "gpt",
                    "value": "| defect property | defect value |\n" +
                             "| --- | --- |\n" +
                             ''.join(["| %s | %s |\n" % (attributes[idx], bool(int(gt_info[attribute]))) for idx,attribute in enumerate(attributes)])

                },
            ]
        # 修改输入描述，对应图片为整体图片，signboard使用red box标注显示
        elif description == 3.5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease describe the defect of the signboard within the red box in this image in table format."
                },
                {
                    "from": "gpt",
                    "value": "| defect property | defect value |\n" +
                             "| --- | --- |\n" +
                             ''.join(["| %s | %s |\n" % (attributes[idx], bool(int(gt_info[attribute]))) for idx,attribute in enumerate(attributes)])

                },
            ]
        elif description == 4:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the entered signboard has any defects."
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect"%get_defect_info(gt_info, attributes, attributes),

                },
            ]
        elif description == 4.5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the signboard in the red box has any defects."
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes),

                },
            ]
        elif description == 5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the entered signboard has any of the following defects:%s"%get_defect_name(attributes)
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes),

                },
            ]
        elif description == 5.5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the signboard in the red polygon has any of the following defects:%s"%get_defect_name(attributes)
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes),

                },
            ]
        img_infos.append(img_info)
    return img_infos


def mdet2llava(img_dir, gt_dir, dst_json, train_ratio=1.0, ref_path=None, description=1):

    os.makedirs(img_dir, exist_ok=True)
    js_data = []

    gt_list = os.listdir(gt_dir)
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        img_infos = get_img_info_mdet(gt_path, img_dir, description)
        js_data += img_infos

    if ref_path is None:
        np.random.seed(0)
        np.random.shuffle(js_data)
        if train_ratio < 1.0:
            train_data_len = int(train_ratio * len(js_data))
            train_data = js_data[0:train_data_len]
            val_data = js_data[train_data_len:]
        else:
            train_data = js_data
            val_data = []
    else:
        ref_list = get_ref_list(ref_path)
        train_data, val_data = [],[]
        for single_data in js_data:
            if single_data['id'].split('_')[0] in ref_list:
                train_data.append(single_data)
            else:
                val_data.append(single_data)

    print('train data len:', len(train_data), 'val data len:', len(val_data))
    with open(dst_json.replace('.json', '_train.json'), 'w') as f:
        output_json = json.dumps(train_data)
        f.write(output_json)
    with open(dst_json.replace('.json', '_val.json'), 'w') as f:
        output_json = json.dumps(val_data)
        f.write(output_json)
# endregion


# region transform
def find_with_defect(input_dir, crop_map_dict):
    if not isinstance(crop_map_dict, dict):
        with open(crop_map_dict, 'r') as f:
            crop_map_dict = json.load(f)
    result_path = input_dir+'.csv'
    defect_dict = {}
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, header=0, index_col=None)
        with_defect = df['value'].any()
        if with_defect:
            src_name = crop_map_dict[Path(file_name).stem + '.jpg']
            if src_name in defect_dict:
                defect_dict[src_name] += 1
            else:
                defect_dict[src_name] = 1
    df_defect = pd.DataFrame(list(defect_dict.items()), columns=['file_name', 'defect_mask_count'])
    df_defect.to_csv(result_path, index=False)
    return df_defect

def copy_mdet_by_llava(label_dir, llava_dir, label_update_dir, crop_map1_path, crop_map2_path):
    with open(crop_map1_path, 'r') as f:
        crop_map_dict = json.load(f)
    with open(crop_map2_path, 'r') as f:
        crop_map_dict_revert = json.load(f)

    print('find defect...')
    result_path = llava_dir+'.csv'
    if not os.path.exists(result_path):
        df_defect = find_with_defect(llava_dir, crop_map_dict)
    else:
        df_defect = pd.read_csv(result_path, header=0, index_col=None)
    print('finished!\n')
    os.makedirs(label_update_dir, exist_ok=True)


    file_with_defect_list = df_defect['file_name'].to_list()
    for file_name in tqdm(file_with_defect_list):
        label_name = Path(file_name).stem + '.txt'
        label_path_src = os.path.join(label_dir, label_name)
        label_path_dst = os.path.join(label_update_dir, label_name)
        with open(label_path_src, 'r') as fr:
            lines = fr.readlines()

        label_llava_list = crop_map_dict_revert[file_name]
        new_lines = []
        with open(label_path_dst, 'w') as fw:
            for obj_id in range(len(lines)):
                file_name_obj = file_name.replace('.jpg', f'_{obj_id}.jpg')
                if file_name_obj not in label_llava_list:
                    print(file_name_obj, 'not exist!')
                else:
                    defect_pred_list = [4] + [0, 0, 0, 0]
                    defect_str = ' '.join(map(str, defect_pred_list)) + ' '
                    obj_line = lines[obj_id]
                    obj_line_new = obj_line[:2] + defect_str + obj_line[2:]
                    new_lines.append(obj_line_new)
            fw.writelines(new_lines)

def update_mdet_by_llava(label_dir, llava_dir, label_update_dir, crop_map1_path, crop_map2_path):
    with open(crop_map1_path, 'r') as f:
        crop_map_dict = json.load(f)
    with open(crop_map2_path, 'r') as f:
        crop_map_dict_revert = json.load(f)

    print('find defect...')
    result_path = llava_dir+'.csv'
    if not os.path.exists(result_path):
        df_defect = find_with_defect(llava_dir, crop_map_dict)
    else:
        df_defect = pd.read_csv(result_path, header=0, index_col=None)
    print('finished!\n')
    os.makedirs(label_update_dir, exist_ok=True)

    file_with_defect_list = df_defect['file_name'].to_list()
    for file_name in tqdm(file_with_defect_list):
        label_name = Path(file_name).stem + '.txt'
        label_path_src = os.path.join(label_dir, label_name)
        label_path_dst = os.path.join(label_update_dir, label_name)
        with open(label_path_src, 'r') as fr:
            lines = fr.readlines()


        label_llava_list = crop_map_dict_revert[file_name]
        new_lines = []
        with open(label_path_dst, 'w') as fw:
            for obj_id in range(len(lines)):
                file_name_obj = file_name.replace('.jpg', f'_{obj_id}.jpg')
                if file_name_obj not in label_llava_list:
                    print(file_name_obj, 'not exist!')
                else:
                    llava_pred_path = os.path.join(llava_dir, Path(file_name_obj).stem + '.txt')
                    df = pd.read_csv(llava_pred_path, header=0, index_col=None)
                    defect_pred_list = [4] + df['value'].astype(int).tolist()
                    defect_str = ' '.join(map(str, defect_pred_list)) + ' '
                    obj_line = lines[obj_id]
                    obj_line_new = obj_line[:2] + defect_str + obj_line[2:]
                    new_lines.append(obj_line_new)
            fw.writelines(new_lines)

def copy_images(image_dir, image_update_dir, label_update_dir):
    os.makedirs(image_update_dir, exist_ok=True)

    label_list = os.listdir(label_update_dir)
    for label_name in tqdm(label_list):
        image_name = Path(label_name).stem + '.jpg'
        image_path_src = os.path.join(image_dir, image_name)
        image_path_dst = os.path.join(image_update_dir, image_name)
        shutil.copy(image_path_src, image_path_dst)

# endregion
if __name__ == '__main__':
    pass

    # # region mdetection data 2 llava
    # root_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001'
    # image_folder = os.path.join(root_dir, 'images')
    # label_folder = os.path.join(root_dir, 'labels')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    #
    # llava_folder = os.path.join(root_dir, 'llava_data')
    # crop_folder = os.path.join(llava_folder, 'images_crop')
    # caption_folder = os.path.join(llava_folder, 'caption')
    # os.makedirs(crop_folder, exist_ok=True)
    # os.makedirs(caption_folder, exist_ok=True)
    # llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    # # region generating dataset
    # # myolo_crop(image_folder, label_folder, crop_folder, class_file,
    # #            attribute_file=attribute_file, seg=True,
    # #            save_method='all',
    # #            crop_method='without_background_box_shape')
    #
    # # mdet2llava(crop_folder, label_folder, llava_caption5_crop,  description=5)
    # # endregion

    # region postprocess dataset
    # root_dir = r'/data/huilin/data/isds/ps_data/0527'
    # image_dir = os.path.join(root_dir, 'images')
    # label_dir = os.path.join(root_dir, 'images_seg_infer', 'labels')
    # result_dir = os.path.join(root_dir, 'images_crop_box_infer5')
    # label_update_dir = os.path.join(root_dir, 'images_crop_box_infer5_updated')
    # image_update_dir = os.path.join(root_dir, 'images_updated')
    # crop_map1_path = os.path.join(root_dir, 'images_crop_box.json')
    # crop_map2_path = os.path.join(root_dir, 'images_crop_box_revert.json')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')

    # update_mdet_by_llava(label_dir, result_dir, label_update_dir, crop_map1_path, crop_map2_path)

    # copy_images(image_dir, image_update_dir, label_update_dir)
    # yolo_mdet_vis(image_dir, label_update_dir, label_update_dir+'_vis',
    #               class_file=class_file,
    #               attribute_file=attribute_file,
    #               seg=True)
    # endregion


    # region mdetection data 2 llava
    # root_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001'
    # image_folder = os.path.join(root_dir, 'images')
    # label_folder = os.path.join(root_dir, 'labels')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    #
    # llava_folder = os.path.join(root_dir, 'llava_data')
    # crop_folder = os.path.join(llava_folder, 'images_crop')
    # caption_folder = os.path.join(llava_folder, 'caption')
    # os.makedirs(crop_folder, exist_ok=True)
    # os.makedirs(caption_folder, exist_ok=True)
    # llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    # region generating dataset
    # myolo_crop(image_folder, label_folder, crop_folder, class_file,
    #            attribute_file=attribute_file, seg=True,
    #            save_method='all',
    #            crop_method='without_background_box_shape')

    # mdet2llava(crop_folder, label_folder, llava_caption5_crop,  description=5)
    # endregion
    root_dir = r'/data/huilin/data/isds/ps_data/0606'
    image_folder = os.path.join(root_dir, 'merge_dir')
    yolo_infer_folder = os.path.join(root_dir, 'merge_dir_seg_infer', 'labels')
    crop_folder = os.path.join(root_dir, 'merge_dir_crop')
    crop_map_path = os.path.join(root_dir, 'images_crop_box.json')
    crop_map_revert_path = os.path.join(root_dir, 'merge_dir_crop_revert.json')
    crop_infer_folder = os.path.join(root_dir, 'merge_dir_crop_risk_infer')
    caption_folder = os.path.join(root_dir, 'caption')
    class_file = os.path.join(root_dir, 'class.txt')
    llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    find_with_defect(crop_infer_folder, crop_map_path)