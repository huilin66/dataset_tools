import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from data_vis.yolo_vis import yolo_mdet_vis
from pathlib import Path
from yolo_mask_crop import myolo_crop
categories = ['background', 'wall frame', 'wall display', 'projecting frame', 'projecting display', 'hanging frame', 'hanging display', 'other']
attributes = ['deformation', 'broken', 'abandonment', 'corrosion']


def get_ref_list(csv_path):
    df = pd.read_csv(csv_path, header=None, index_col=None, names=['path'])
    file_path_list = df['path'].to_list()
    file_name_list = [Path(os.path.basename(file_path)).stem for file_path in file_path_list]
    return file_name_list

def cp_imgs(img_dir, dst_img_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    for category in categories:
        category_dir = os.path.join(img_dir, category, 'all')
        if not os.path.exists(category_dir):
            continue
        img_list = os.listdir(category_dir)
        for img_name in tqdm(img_list):
            input_path = os.path.join(category_dir, img_name)
            output_path = os.path.join(dst_img_dir, img_name)
            shutil.copyfile(input_path, output_path)

def get_gt_info(gt_path):
    def get_level(df):
        if len(df) > 2:
            return 'serious'
        elif len(df) == 2:
            if df['area'].max() < 0.1 and df['w_box'].max() < 0.1 and df['h_box'].max() < 0.1:
                return 'moderate'
            else:
                return 'serious'
        else:
            if df['area'].max() < 0.1 and df['w_box'].max() < 0.1 and df['h_box'].max() < 0.1:
                return 'slight'
            else:
                return 'serious'
    gt_info = {}
    df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ',
                     names=['cat_id', 'x_center', 'y_center', 'w_box', 'h_box'])
    df['area'] = df['w_box']*df['h_box']
    cat_list = []
    for idx,row in df.iterrows():
        cat_name = id2cat_map[row['cat_id']]
        cat_list.append(cat_name)
    cat_set = list(set(cat_list))
    gt_info['type'] = ';'.join(cat_set)
    gt_info['number'] = len(df)
    gt_info['level'] = get_level(df)
    causes,actions = [],[]
    for cat_name in cat_list:
        cat_id = cat2id_map[cat_name]
        cause = id2cause_map[cat_id]
        action = id2action_map[cat_id]
        causes.append(cause)
        actions.append(action)
    gt_info['cause'] = causes[0] #';'.join(causes)
    gt_info['action'] = actions[0] #';'.join(actions)
    return gt_info


def get_img_info(img_path, gt_path):
    gt_info = get_gt_info(gt_path)
    img_info = {}
    img_info['id'] = os.path.basename(img_path).replace('.jpg', '')
    img_info['image'] = 'defect/'+ os.path.basename(img_path)
    img_info['conversations'] = [
        {
            "from": "human",
            "value": "<image>\nplease describe this image in table format."
        },
        {
            "from": "gpt",
            "value": "| property | value |\n"
                     "| --- | --- |\n"
                     "| background | %s |\n"
                     "| defect types | %s |\n"
                     "| defect numbers | %s |\n"
                     "| defect level | %s |\n"
                     "| possible causes of the defects | %s |\n"
                     "| required actions of the defects| %s |"%
                     ('road', 'crack', gt_info['number'], gt_info['level'], gt_info['cause'], gt_info['action'])
        },
    ]
    if gt_info['type'] != 'background':
        print(gt_info['type'])
    return img_info


def det2llava(img_dir, gt_dir, dst_json):
    js_data = []
    img_list = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]
    gt_list = [img_path.replace(img_dir, gt_dir).replace('.jpg', '.txt') for img_path in img_list]
    for idx in tqdm(range(len(img_list))):
        img_path, gt_path = img_list[idx], gt_list[idx]
        img_info = get_img_info(img_path, gt_path)
        js_data.append(img_info)
    with open(dst_json, 'w') as f:
        output_json = json.dumps(js_data)
        f.write(output_json)


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

def prediction_num_check(img_dir, predict_dir):
    img_list = os.listdir(img_dir)
    for img_name in tqdm(img_list):
        predict_path = os.path.join(predict_dir, Path(img_name).stem+'.txt')
        if not os.path.exists(predict_path):
            with open(predict_path, 'w') as f:
                pass  # 不写入任何内容


def mdetresult2llava(img_dir, vis_img_dir, crop_img_dir, predict_dir, class_file, attribute_file, crop_keep_shape=False, det_crop=False):
    prediction_num_check(img_dir, predict_dir)
    yolo_mdet_vis(img_dir, predict_dir, vis_img_dir, class_file, crop_dir=crop_img_dir, seg=False,
                  attribute_file=attribute_file, filter_no=True, crop_keep_shape=crop_keep_shape, det_crop=det_crop)

    final_img_dir = os.path.join(crop_img_dir, 'final')
    os.makedirs(final_img_dir, exist_ok=True)
    for category in categories:
        category_dir = os.path.join(crop_img_dir, category, 'all')
        if not os.path.exists(category_dir):
            continue
        img_list = os.listdir(category_dir)
        for img_name in tqdm(img_list):
            input_path = os.path.join(category_dir, img_name)
            output_path = os.path.join(final_img_dir, img_name)
            shutil.copyfile(input_path, output_path)

def llavaresult2mdet(input_dir, output_dir, ref_label_dir, attributes):
    os.makedirs(output_dir, exist_ok=True)
    label_list = os.listdir(ref_label_dir)
    for label_name in tqdm(label_list):
        label_path = os.path.join(ref_label_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        if os.stat(label_path).st_size == 0:
            pass
        else:
            df = pd.read_csv(label_path, header=None, index_col=None, sep=' ',
                             names=['cat_id', len(attributes)] + attributes + ['x_center', 'y_center', 'w_box', 'h_box', 'conf'])
            for idx in range(len(df)):
                input_path = os.path.join(input_dir, label_name.replace('.txt', '_%d.txt' % idx))
                if not os.path.exists(input_path) or os.stat(input_path).st_size == 0:
                    for attribute in attributes:
                        df.loc[idx, attribute] = 0
                else:
                    df_input = pd.read_csv(input_path, index_col='property')
                    df_input = df_input['value']
                    for attribute in attributes:
                        df.loc[idx, attribute] = int(bool(df_input[attribute]))
            df.to_csv(output_path, header=None, index=None, sep=' ')


def mdet_val(predict_dir, label_dir):
    def calculate_iou(box_pred, box_true):
        # box format: [x1, y1, x2, y2]
        x1_pred, y1_pred, x2_pred, y2_pred = box_pred
        x1_true, y1_true, x2_true, y2_true = box_true

        # Intersection coordinates
        x1_inter = max(x1_pred, x1_true)
        y1_inter = max(y1_pred, y1_true)
        x2_inter = min(x2_pred, x2_true)
        y2_inter = min(y2_pred, y2_true)

        # Intersection area
        intersection_width = max(0, x2_inter - x1_inter)
        intersection_height = max(0, y2_inter - y1_inter)
        intersection_area = intersection_width * intersection_height

        # Areas of the predicted and true boxes
        area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        area_true = (x2_true - x1_true) * (y2_true - y1_true)

        # Union area
        union_area = area_pred + area_true - intersection_area

        # Compute IOU
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def get_tp_fp_fn(pred_boxes, true_boxes, iou_threshold=0.5):
        tp, fp = 0, 0
        detected_true_boxes = []

        for pred in pred_boxes:
            max_iou = 0
            best_true_idx = -1
            for idx, true_box in enumerate(true_boxes):
                iou = calculate_iou(pred, true_box)
                if iou > max_iou:
                    max_iou = iou
                    best_true_idx = idx

            if max_iou >= iou_threshold and best_true_idx not in detected_true_boxes:
                tp += 1
                detected_true_boxes.append(best_true_idx)
            else:
                fp += 1

        fn = len(true_boxes) - len(detected_true_boxes)
        return tp, fp, fn

    def calculate_ap(precision, recall):
        # Insert 0 and 1 at the beginning and end of precision and recall
        precision = np.concatenate(([0], precision, [0]))
        recall = np.concatenate(([0], recall, [1]))

        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])

        # Compute area under the precision-recall curve
        ap = 0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i - 1]) * precision[i]

        return ap

    def calculate_map(pred_boxes_all, true_boxes_all, scores_all, iou_threshold=0.5):
        aps = []
        for class_idx in range(len(pred_boxes_all)):  # Loop over classes
            pred_boxes = pred_boxes_all[class_idx]
            true_boxes = true_boxes_all[class_idx]
            scores = scores_all[class_idx]

            # Sort predictions by score (confidence)
            sorted_indices = np.argsort(-scores)
            pred_boxes = [pred_boxes[i] for i in sorted_indices]

            tp, fp, fn = [], [], []
            for pred_box_set, true_box_set in zip(pred_boxes, true_boxes):
                tp_c, fp_c, fn_c = get_tp_fp_fn(pred_box_set, true_box_set, iou_threshold)
                tp.append(tp_c)
                fp.append(fp_c)

            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)

            ap = calculate_ap(precision, recall)
            aps.append(ap)

        return np.mean(aps)

    # # Example usage:
    # # pred_boxes_all, true_boxes_all, scores_all are lists where each element represents a class,
    # # and within each class, they contain the predicted and true boxes with corresponding confidence scores.
    # # pred_boxes_all = [...]
    # # true_boxes_all = [...]
    # # scores_all = [...]
    #
    # map_50 = calculate_map(pred_boxes_all, true_boxes_all, scores_all, iou_threshold=0.5)
    # print(f"mAP@50: {map_50}")

    label_list = os.listdir(label_dir)
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        predict_path = os.path.join(predict_dir, label_name)
        df_label = pd.read_csv(predict_path, header=None, index_col=None, sep=' ',)
        df_predict = pd.read_csv(predict_path, header=None, index_col=None, sep=' ',)

def split_trainval(img_dir, ref_path, get_train=False, get_val=False):
    if get_train:
        img_dir_train = img_dir+'_train'
        os.makedirs(img_dir_train, exist_ok=True)
    if get_val:
        img_dir_val = img_dir+'_val'
        os.makedirs(img_dir_val, exist_ok=True)

    train_list = get_ref_list(ref_path)
    file_list = os.listdir(img_dir)
    for img_name in tqdm(file_list):
        img_prefix = img_name.split('_')[0]
        img_path = os.path.join(img_dir, img_name)
        if img_prefix in train_list:
            if get_train:
                dst_path = os.path.join(img_dir_train, img_name)
                shutil.copy(img_path, dst_path)
        else:
            if get_val:
                dst_path = os.path.join(img_dir_val, img_name)
                shutil.copy(img_path, dst_path)

if __name__ == '__main__':
    pass

    # region mdetection data 2 llava
    root_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001'
    image_folder = os.path.join(root_dir, 'images')
    label_folder = os.path.join(root_dir, 'labels')
    attribute_file = os.path.join(root_dir, 'attribute.yaml')
    class_file = os.path.join(root_dir, 'class.txt')

    llava_folder = os.path.join(root_dir, 'llava_data')
    crop_folder = os.path.join(llava_folder, 'images_crop')
    caption_folder = os.path.join(llava_folder, 'caption')
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(caption_folder, exist_ok=True)
    llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    # region generating dataset
    # myolo_crop(image_folder, label_folder, crop_folder, class_file,
    #            attribute_file=attribute_file, seg=True,
    #            save_method='all',
    #            crop_method='without_background_box_shape')

    mdet2llava(crop_folder, label_folder, llava_caption5_crop,  description=5)
    # endregion
