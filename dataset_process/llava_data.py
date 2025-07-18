import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from data_vis.yolo_vis import yolo_mdet_vis
from yolo_eval import yolo_mdet_eval, yolo_remove_conf
from yolo_split import copy_split
from pathlib import Path

categories = ['background', 'projecting_signboard', 'wall_signboard']
attributes = ['surface_missing', 'surface_incomplete', 'surface_corroded', 'frame_corroded', 'surface_peeling',
              'surface_fade', 'surface_deformed', 'frame_deformed', 'disconnected', 'added_billboard']
attributes2 = [attribute.replace('_', ' ') for attribute in attributes]
attributes_name = ['missing surface', 'incomplete surface', 'corroded surface', 'corroded frame', 'peeling surface',
                   'faded surface', 'deformed surface', 'deformed frame', 'disconnected', 'unauthorized']
id2cat_map = {
    0: 'background',
    1: 'crack',
    2: 'concrete_spalling',
    3: 'finishes_peeling',
    4: 'mold',
    5: 'water_seepage',
}
cat2id_map = {
    'background':0,
    'crack':1,
    'concrete_spalling':2,
    'finishes_peeling':3,
    'mold':4,
    'water_seepage':5,
}
id2cause_map = {
    # 0: 'background',
    0: 'heavy traffic, temperature changes, and other factors.',
    1: 'shrink age or contraction of the rendering on the walls or poor workmanship during construction.',
    2: 'ageing buildings; prolonged seepage of water damages the steel bars inside the concrete; steel bars become rusty.',
    3: 'ageing, structural movements, poor workmanship during installation thermal movement, inadequate expansion joints damage by external factor, ingress of water into the gap between the finishes and the surface of the wall.',
    4: 'Moisture from wet areas, vandalism or accidents, impacts from occupants or loads, deteriorates faster than expected.',
    5: 'defective fabric or installations of buildings and the lack of proper maintenance, leakage from defective water pipes, sanitary fitments or drainage pipes.',

}

id2action_map = {
    # 0: 'background',
    0: 'Contact the highway department to repair the defect.',
    1: 'Building owners should arrange for timely repair and maintenance works to upkeep the building in good condition.',
    2: 'Building owners should arrange for timely repair and maintenance works to upkeep the building in good condition.',
    3: 'Good management of the building is the key to maintaining building safety. Building owners are therefore advised to appoint competent management companies to manage their buildings.',
    4: 'Building owners should keep in view the conditions of the defects,unless the circumstances have changed, owners may carry out repair as necessary.',
    5: 'owners should investigate the source of seepage by liaising with owners of the flat concerned for carrying out repair works as early as possible,',
}


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
    df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ',
                     names=['cat_id', len(attributes)] + attributes + ['x_center', 'y_center', 'w_box', 'h_box'])
    for idx,row in df.iterrows():
        gt_info = {}
        img_name = os.path.basename(gt_path).replace('.txt', '_%d.jpg' % idx)
        gt_info['img_name'] = img_name
        gt_info['category'] = categories[int(row['cat_id'])]
        for attribute in attributes:
            gt_info[attribute] = row[attribute]
        gt_infos.append(gt_info)
    return gt_infos


def get_img_info_mdet(gt_path, img_dir, description=1):
    def get_defect_name(attributes_names):
        attributes_names_list = ["'%s'"%attribute_name for attribute_name in attributes_names]
        names_str = ' '.join(attributes_names_list)
        return names_str
    def get_defect_info(gt_info, attributes, attributes_name):
        attributes_names_list = []
        for idx, attribute in enumerate(attributes):
            if bool(int(gt_info[attribute])):
                attributes_names_list.append("'%s'"%attributes_name[idx])
        if len(attributes_names_list) == 0:
            return "no"
        else:
            return ' '.join(attributes_names_list)
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
                             ''.join(["| %s | %s |\n" % (attributes_name[idx], bool(int(gt_info[attribute]))) for idx,attribute in enumerate(attributes)])

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
                             ''.join(["| %s | %s |\n" % (attributes_name[idx], bool(int(gt_info[attribute]))) for idx,attribute in enumerate(attributes)])

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
                    "value": "The entered signboard has %s defect"%get_defect_info(gt_info, attributes, attributes_name),

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
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes_name),

                },
            ]
        elif description == 5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the entered signboard has any of the following defects:%s"%get_defect_name(attributes_name)
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes_name),

                },
            ]
        elif description == 5.5:
            img_info['conversations'] = [
                {
                    "from": "human",
                    "value": "<image>\nPlease check if the signboard in the red box has any of the following defects:%s"%get_defect_name(attributes_name)
                },
                {
                    "from": "gpt",
                    "value": "The entered signboard has %s defect" % get_defect_info(gt_info, attributes, attributes_name),

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
    # region detection data 2 llava

    # img_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\images\train'
    # gt_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\labels\train'
    # dst_json = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect2k.json'
    # det2llava(img_dir, gt_dir, dst_json)

    # img_dir = r'E:\data\2023_defect\road_crack_detection.v2i.yolov9\train\images'
    # gt_dir = r'E:\data\2023_defect\road_crack_detection.v2i.yolov9\train\labels'
    # dst_json = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect1k.json'
    # det2llava(img_dir, gt_dir, dst_json)

    # data1 = json.load(open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect1k.json'))
    # data2 = json.load(open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect2k.json'))
    #
    # data = data1+data2
    # print(len(data1), len(data2), len(data))
    # with open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect4k.json', 'w') as f:
    #     output_json = json.dumps(data)
    #     f.write(output_json)

    # endregion


    # region mdetection data 2 llava

    root_dir_mdet = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c'
    image_folder = os.path.join(root_dir_mdet, 'images')
    label_folder = os.path.join(root_dir_mdet, 'labels')
    vis_folder = os.path.join(root_dir_mdet, 'images_vis')
    crop_folder1 = os.path.join(root_dir_mdet, 'images_crop')
    crop_folder2 = os.path.join(root_dir_mdet, 'images_crop_keep')
    crop_folder3 = os.path.join(root_dir_mdet, 'images_crop_det')
    infer_vis_folder = os.path.join(root_dir_mdet, 'infer_images_vis')
    infer_crop_folder1 = os.path.join(root_dir_mdet, 'infer_images_crop')
    infer_crop_folder2 = os.path.join(root_dir_mdet, 'infer_images_crop_keep')
    infer_crop_folder3 = os.path.join(root_dir_mdet, 'infer_images_crop_det')
    image_val_folder = os.path.join(root_dir_mdet, 'images_val')
    label_val_folder = os.path.join(root_dir_mdet, 'labels_val')
    mayolo_infer1_folder = os.path.join(root_dir_mdet, 'mayolo_infer1')
    mayolo_infer2_folder = os.path.join(root_dir_mdet, 'mayolo_infer2')
    mayolo_infer2_folder_withoutconf = os.path.join(root_dir_mdet, 'mayolo_infer2_withoutconf')
    attribute_file = os.path.join(root_dir_mdet, 'attribute.yaml')
    class_file = os.path.join(root_dir_mdet, 'class.txt')
    train_csv_path = os.path.join(root_dir_mdet, 'train.txt')
    val_csv_path = os.path.join(root_dir_mdet, 'val.txt')

    root_dir_llava = dst_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava'
    img_det_folder_llava = os.path.join(root_dir_llava, 'images_det')
    img_crop_folder_llava = os.path.join(root_dir_llava, 'images_crop')
    img_keep_folder_llava = os.path.join(root_dir_llava, 'images_keep')
    img_infer_det_folder_llava = os.path.join(root_dir_llava, 'infer_images_det')
    img_infer_crop_folder_llava = os.path.join(root_dir_llava, 'infer_images_crop')
    img_infer_keep_folder_llava = os.path.join(root_dir_llava, 'infer_images_keep')
    caption_folder = os.path.join(dst_dir, 'caption')
    llava_caption1_crop = os.path.join(caption_folder, 'signboard_caption1_crop.json')
    llava_caption1_keep = os.path.join(caption_folder, 'signboard_caption1_keep.json')
    llava_caption2_crop = os.path.join(caption_folder, 'signboard_caption2_crop.json')
    llava_caption2_keep = os.path.join(caption_folder, 'signboard_caption2_keep.json')
    llava_caption3_crop = os.path.join(caption_folder, 'signboard_caption3_crop.json')
    llava_caption3_keep = os.path.join(caption_folder, 'signboard_caption3_keep.json')
    llava_caption3_det = os.path.join(caption_folder, 'signboard_caption3_det.json')
    llava_caption4_crop = os.path.join(caption_folder, 'signboard_caption4_crop.json')
    llava_caption4_keep = os.path.join(caption_folder, 'signboard_caption4_keep.json')
    llava_caption4_det = os.path.join(caption_folder, 'signboard_caption4_det.json')
    llava_caption5_crop = os.path.join(caption_folder, 'signboard_caption5_crop.json')
    llava_caption5_keep = os.path.join(caption_folder, 'signboard_caption5_keep.json')
    llava_caption5_det = os.path.join(caption_folder, 'signboard_caption5_det.json')
    llava_infer_results_folder = os.path.join(root_dir_llava, 'llava_images_infer')
    llava_infer_mdetect_folder = os.path.join(root_dir_llava, 'llava_images_infer_mdetect_format')

    # region step1 generating dataset

    # copy val image and label
    # copy_split(image_folder, label_folder, image_val_folder, label_val_folder, val_csv_path)

    # generate image data in mdetect dataset
    # yolo_mdet_vis(image_folder, label_folder, vis_folder, class_file, crop_dir=crop_folder1, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=False, det_crop=False)
    # yolo_mdet_vis(image_folder, label_folder, vis_folder, class_file, crop_dir=crop_folder2, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=True, det_crop=False)
    # yolo_mdet_vis(image_folder, label_folder, vis_folder, class_file, crop_dir=crop_folder3, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=True, det_crop=True)

    # move image data from mdetect to llava
    # cp_imgs(crop_folder1, img_crop_folder_llava)
    # cp_imgs(crop_folder2, img_keep_folder_llava)
    # cp_imgs(crop_folder3, img_det_folder_llava)

    # generate caption dataset
    # os.makedirs(caption_folder, exist_ok=True)
    # mdet2llava(img_crop_folder_llava, label_folder, llava_caption1_crop, ref_path=train_csv_path, description=1)
    # mdet2llava(img_keep_folder_llava, label_folder, llava_caption1_keep, ref_path=train_csv_path, description=1)
    # mdet2llava(img_crop_folder_llava, label_folder, llava_caption2_crop, ref_path=train_csv_path, description=2)
    # mdet2llava(img_keep_folder_llava, label_folder, llava_caption2_keep, ref_path=train_csv_path, description=2)
    # mdet2llava(img_crop_folder_llava, label_folder, llava_caption3_crop, ref_path=train_csv_path, description=3)
    # mdet2llava(img_keep_folder_llava, label_folder, llava_caption3_keep, ref_path=train_csv_path, description=3)
    # mdet2llava(img_det_folder_llava, label_folder, llava_caption3_det, ref_path=train_csv_path, description=3.5)
    # mdet2llava(img_crop_folder_llava, label_folder, llava_caption4_crop, ref_path=train_csv_path, description=4)
    # mdet2llava(img_keep_folder_llava, label_folder, llava_caption4_keep, ref_path=train_csv_path, description=4)
    # mdet2llava(img_det_folder_llava, label_folder, llava_caption4_det, ref_path=train_csv_path, description=4.5)
    # mdet2llava(img_crop_folder_llava, label_folder, llava_caption5_crop, ref_path=train_csv_path, description=5)
    # mdet2llava(img_keep_folder_llava, label_folder, llava_caption5_keep, ref_path=train_csv_path, description=5)
    # mdet2llava(img_det_folder_llava, label_folder, llava_caption5_det, ref_path=train_csv_path, description=5.5)

    # get val images
    # split_trainval(img_crop_folder_llava, ref_path=train_csv_path, get_train=False, get_val=True)
    # split_trainval(img_keep_folder_llava, ref_path=train_csv_path, get_train=False, get_val=True)
    # split_trainval(img_det_folder_llava, ref_path=train_csv_path, get_train=False, get_val=True)

    # calculate map, oa metric
    # yolo_mdet_eval(mayolo_infer1_folder, label_val_folder, conf=0.0, with_prob=True)
    # yolo_mdet_eval(mayolo_infer2_folder, label_val_folder, conf=0.0, with_prob=True)

    # endregion

    # region step2 convert mayolo prediction result into llava data


    # remove conf for vis and crop
    # yolo_remove_conf(mayolo_infer2_folder, mayolo_infer2_folder_withoutconf, mdet=True)

    # generate image data for mayolo infer result
    # yolo_mdet_vis(image_val_folder, mayolo_infer2_folder_withoutconf, infer_vis_folder, class_file, crop_dir=infer_crop_folder1, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=False, det_crop=False)
    # yolo_mdet_vis(image_val_folder, mayolo_infer2_folder_withoutconf, infer_vis_folder, class_file, crop_dir=infer_crop_folder2, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=True, det_crop=False)
    # yolo_mdet_vis(image_val_folder, mayolo_infer2_folder_withoutconf, infer_vis_folder, class_file, crop_dir=infer_crop_folder3, seg=False,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=True, det_crop=True)

    # move image data from mdetect to llava
    # cp_imgs(infer_crop_folder1, img_infer_crop_folder_llava)
    # cp_imgs(infer_crop_folder2, img_infer_keep_folder_llava)
    # cp_imgs(infer_crop_folder3, img_infer_det_folder_llava)



    # convert llava infer result 2 mdetect format
    # for infer_name in os.listdir(llava_infer_results_folder):
    #     print(infer_name)
    #     infer_path = os.path.join(llava_infer_results_folder, infer_name)
    #     result_path = os.path.join(llava_infer_mdetect_folder, infer_name)
    #     if '1' in infer_name:
    #         llavaresult2mdet(infer_path, result_path, mayolo_infer2_folder, attributes)
    #     elif '2' in infer_name:
    #         llavaresult2mdet(infer_path, result_path, mayolo_infer2_folder, attributes2)
    #     else:
    #         llavaresult2mdet(infer_path, result_path, mayolo_infer2_folder, attributes_name)


    # get the infer metrics
    for infer_name in os.listdir(llava_infer_results_folder):
        print(f"{infer_name} result:")
        result_path = os.path.join(llava_infer_mdetect_folder, infer_name)
        yolo_mdet_eval(result_path, label_val_folder, conf=0.0, with_prob=True)
        print()
    # endregion

    # endregion

    # region old code
    # dst_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava_infer'
    # gt_dir = os.path.join(root_dir, 'labels')
    # infer_dir = os.path.join(root_dir, 'mayolo_infer')
    # train_csv_path = os.path.join(root_dir, 'train.txt')

    # src_img_dir = os.path.join(root_dir, 'images_crop')
    # dst_img_dir = os.path.join(dst_dir, 'images')
    # dst_img_dir_train = os.path.join(dst_dir, 'images_train')
    # dst_img_dir_val = os.path.join(dst_dir, 'images_val')
    # cp_imgs(src_img_dir, dst_img_dir)
    # split_trainval(dst_img_dir, dst_img_dir_train, dst_img_dir_val, ref_path=train_csv_path)
    #
    # src_img_dir = os.path.join(root_dir, 'images_crop_keep')
    # dst_img_dir = os.path.join(dst_dir, 'images_keep')
    # dst_img_dir_train = os.path.join(dst_dir, 'images_keep_train')
    # dst_img_dir_val = os.path.join(dst_dir, 'images_keep_val')
    # cp_imgs(src_img_dir, dst_img_dir)
    # split_trainval(dst_img_dir, dst_img_dir_train, dst_img_dir_val, ref_path=train_csv_path)
    #
    # src_img_dir = os.path.join(root_dir, 'images_crop_det')
    # dst_img_dir = os.path.join(dst_dir, 'images_det')
    # dst_img_dir_train = os.path.join(dst_dir, 'images_det_train')
    # dst_img_dir_val = os.path.join(dst_dir, 'images_det_val')
    # cp_imgs(src_img_dir, dst_img_dir)
    # split_trainval(dst_img_dir, dst_img_dir_train, dst_img_dir_val, ref_path=train_csv_path)


    # dst_img_dir = os.path.join(root_dir, 'images')
    # dst_json = os.path.join(dst_dir, 'signboard_caption1.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=1)
    #
    # dst_img_dir = os.path.join(root_dir, 'images_keep')
    # dst_json = os.path.join(dst_dir, 'signboard_caption1_keep.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=1)
    #
    # dst_img_dir = os.path.join(root_dir, 'images')
    # dst_json = os.path.join(dst_dir, 'signboard_caption2.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=2)
    #
    # dst_img_dir = os.path.join(root_dir, 'images_keep')
    # dst_json = os.path.join(dst_dir, 'signboard_caption2_keep.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=2)
    #
    # dst_img_dir = os.path.join(root_dir, 'images')
    # dst_json = os.path.join(dst_dir, 'signboard_caption3.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=3)
    #
    # dst_img_dir = os.path.join(root_dir, 'images_keep')
    # dst_json = os.path.join(dst_dir, 'signboard_caption3_keep.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=3)
    #
    # dst_img_dir = os.path.join(root_dir, 'images_det')
    # dst_json = os.path.join(dst_dir, 'signboard_caption3_det.json')
    # mdet2llava(dst_img_dir, gt_dir, dst_json, ref_path=train_csv_path, description=3.5)


    # root_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c'
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    # img_dir = os.path.join(root_dir, 'images_val')
    # vis_img_dir = os.path.join(root_dir, 'images_infer_vis')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    # predict_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\mayolo_infer'

    # crop_img_dir = os.path.join(root_dir, 'images_infer_crop')
    # dst_img_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_crop'
    # mdetresult2llava(img_dir, vis_img_dir, crop_img_dir, predict_dir, class_file, attribute_file)
    # cp_imgs(crop_img_dir, dst_img_dir)
    # crop_img_dir = os.path.join(root_dir, 'images_infer_keep')
    # dst_img_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_keep'
    # mdetresult2llava(img_dir, vis_img_dir, crop_img_dir, predict_dir, class_file, attribute_file, crop_keep_shape=True)
    # cp_imgs(crop_img_dir, dst_img_dir)
    # crop_img_dir = os.path.join(root_dir, 'images_infer_det')
    # dst_img_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_det'
    # mdetresult2llava(img_dir, vis_img_dir, crop_img_dir, predict_dir, class_file, attribute_file, crop_keep_shape=True, det_crop=True)
    # cp_imgs(crop_img_dir, dst_img_dir)

    # root_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c'
    # infer_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_result'
    # mdet_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_result_mdet'
    # predict_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\mayolo_infer'
    # for infer_name in os.listdir(infer_dir):
    #     print(infer_name)
    #     infer_path = os.path.join(infer_dir, infer_name)
    #     result_path = os.path.join(mdet_dir, infer_name)
    #     if '1' in infer_name:
    #         llavaresult2mdet(infer_path, result_path, predict_dir, attributes)
    #     elif '2' in infer_name:
    #         llavaresult2mdet(infer_path, result_path, predict_dir, attributes2)
    #     else:
    #         llavaresult2mdet(infer_path, result_path, predict_dir, attributes_name)
    # endregion