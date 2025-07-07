import os
from pathlib import Path

import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


def compute_iou_box(box1, box2):
    '''
    x1, y1, x2, y2
    '''
    x1_tp, y1_tp, x1_bt, y1_bt = box1
    x2_tp, y2_tp, x2_bt, y2_bt = box2
    x1 = max(x1_tp, x2_tp)
    y1 = max(y1_tp, y2_tp)
    x2 = min(x1_bt, x2_bt)
    y2 = min(y1_bt, y2_bt)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def poly2xywh(polygon):
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    x = (x_min + x_max)/2
    y = (y_min + y_max)/2
    width = x_max - x_min
    height = y_max - y_min
    return [x, y, width, height]

def parse_boxes(file_path, image_width, image_height, pred=False, with_att=False, with_conf=False):
    boxes = []
    df = pd.DataFrame(None, columns=['class', 'x1', 'y1', 'x2', 'y2', 'conf'])
    atts = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            cls = int(parts[0])
            if with_att:
                att_len = int(parts[1])
                att = list(map(int, parts[2:2+att_len]))
                if pred:
                    att = [1 if x>0 else 0 for x in att]
                coord_idx = 2+att_len
            else:
                att = []
                coord_idx = 1
            atts.append(att)

            x = float(parts[coord_idx + 0])
            y = float(parts[coord_idx + 1])
            h = float(parts[coord_idx + 2])
            w = float(parts[coord_idx + 3])
            boxes.append((cls, x, y, h, w))
            if with_conf:
                conf = float(parts[-1])
            else:
                conf = 1
            df.loc[len(df)] = [cls, x, y, h, w, conf]
    if with_att:
        columns = ['att%d'%i for i in range(len(atts[0]))]
        df_att = pd.DataFrame(atts, columns=columns)
        df = pd.concat([df,df_att], axis=1)
    boxes_abs = box2box_abs(boxes, image_width, image_height)
    return boxes_abs, df


def parse_masks(file_path, image_width, image_height, pred=False, with_att=False, with_conf=False):
    boxes = []
    masks = []
    df = pd.DataFrame(None, columns=['id', 'class', 'xy', 'conf'])
    atts = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            cls = int(parts[0])
            if with_att:
                att_len = int(parts[1])
                att = list(map(int, parts[2:2 + att_len]))
                if pred:
                    att = [1 if x > 0 else 0 for x in att]
                coord_idx = 2 + att_len
            else:
                att = []
                coord_idx = 1

            if with_conf:
                conf = float(parts[-1])
                mask = list(map(float, parts[coord_idx:-1]))
            else:
                conf = 1
                mask = list(map(float, parts[coord_idx:]))

            masks.append([cls]+mask)
            atts.append(att)
            df.loc[len(df)] = [idx, cls, mask, conf]
            polygon = np.array(mask).reshape(-1, 2)
            if len(mask) > 1 and len(mask)%2==0:
                box = poly2xywh(polygon)
            else:
                box = [0, 0, 0, 0]
            boxes.append([cls]+box)
    if with_att:
        columns = ['att%d' % i for i in range(len(atts[0]))]
        df_att = pd.DataFrame(atts, columns=columns)
        df = pd.concat([df, df_att], axis=1)
    boxes_abs = box2box_abs(boxes, image_width, image_height)
    masks_abs = mask2mask_abs(masks, image_width, image_height)
    return boxes_abs, masks_abs, df


def box2box_abs(boxes, W, H):
    abs_boxes = []
    for cls, x, y, h, w in boxes:
        x_center = x * W
        y_center = y * H
        w_abs = w * W
        h_abs = h * H
        x1 = x_center - w_abs / 2
        y1 = y_center - h_abs / 2
        x2 = x_center + w_abs / 2
        y2 = y_center + h_abs / 2
        abs_boxes.append((cls, x1, y1, x2, y2))
    return abs_boxes

def mask2mask_abs(masks, W, H):
    abs_masks = []
    for mask in masks:
        class_id = mask.pop(0)
        abs_mask = [class_id]
        for i in range(0, len(mask), 2):
            abs_mask.append(mask[i] * W)
            abs_mask.append(mask[i+1] * H)
        abs_masks.append(abs_mask)
    return abs_masks



def compute_cost_matrix(label_boxes, pred_boxes, threshold=0.5):
    n_labels = len(label_boxes)
    n_preds = len(pred_boxes)
    cost_matrix = np.full((n_labels, n_preds), np.inf)  # 初始化为 inf

    for i, label_box in enumerate(label_boxes):
        for j, pred_box in enumerate(pred_boxes):
            if label_box[0] == pred_box[0]:  # 仅同类匹配
                iou = compute_iou_box(label_box, pred_box)
                if iou >= threshold:
                    cost_matrix[i, j] = -iou  # 最小化 -iou ≈ 最大化 iou

    return cost_matrix

def match_boxes(label_boxes_abs, pred_boxes_abs, save_dir, threshold=0.5):
    num_labels = len(label_boxes_abs)
    num_preds = len(pred_boxes_abs)
    cost_matrix = np.zeros((num_labels, num_preds))+100
    # cost_matrix = compute_cost_matrix(label_boxes_abs, pred_boxes_abs, threshold)

    for i in range(num_labels):
        cls_label, x1_l, y1_l, x2_l, y2_l = label_boxes_abs[i]
        for j in range(num_preds):
            cls_pred, x1_p, y1_p, x2_p, y2_p = pred_boxes_abs[j]
            if cls_label != cls_pred:
                continue
            iou = compute_iou_box((x1_l, y1_l, x2_l, y2_l), (x1_p, y1_p, x2_p, y2_p))
            if iou >= threshold:
                cost_matrix[i][j] = -iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match_dict = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] < 100:
            match_dict[r] = c
    return match_dict

def match_masks(label_masks_abs, pred_masks_abs, save_dir, threshold=0.5):
    num_labels = len(label_masks_abs)
    num_preds = len(pred_masks_abs)
    cost_matrix = np.zeros((num_labels, num_preds))+100


    true_classes = np.array([label_mask[0] for label_mask in label_masks_abs])
    pred_classes = np.array([pred_mask[0] for pred_mask in pred_masks_abs])
    correct_class = true_classes[:, None] == pred_classes

    iou_matrix = np.zeros((num_labels, num_preds))
    true_polygon = [Polygon(np.array(mask[1:]).reshape(-1, 2)) for mask in label_masks_abs]
    pred_polygon = [Polygon(np.array(mask[1:]).reshape(-1, 2)) for mask in pred_masks_abs]
    for i, true_poly in enumerate(true_polygon):
        for j, pred_poly in enumerate(pred_polygon):
            if not true_poly.is_valid:
                true_poly = true_poly.buffer(0)
            if not pred_poly.is_valid:
                pred_poly = pred_poly.buffer(0)
            inter = true_poly.intersection(pred_poly).area
            union = true_poly.union(pred_poly).area
            iou = inter / union if union > 0 else 0.0
            iou_matrix[i][j] = iou
    iou = iou_matrix * correct_class

    iou50 = iou >= 0.5
    correct_mask = correct_class & iou50
    sum = np.sum(correct_mask)


    for i in range(num_labels):
        cls_label, x1_l, y1_l, x2_l, y2_l = label_masks_abs[i]
        for j in range(num_preds):
            cls_pred, x1_p, y1_p, x2_p, y2_p = pred_masks_abs[j]
            if cls_label != cls_pred:
                continue
            iou = compute_iou_box((x1_l, y1_l, x2_l, y2_l), (x1_p, y1_p, x2_p, y2_p))
            if iou >= threshold:
                cost_matrix[i][j] = -iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match_dict = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] < 100:
            match_dict[r] = c
    return match_dict

def match_masks_npy(label_masks, pred_masks, threshold, label_npy_path=None, pred_npy_path=None):
    pass
    num_labels = len(label_masks)
    num_preds = len(pred_masks)
    cost_matrix = np.zeros((num_labels, num_preds))+100

    true_classes = np.load(label_npy_path.replace('labels_npy', 'labels_npy_cls'))
    # true_classes = np.array([label_mask[0] for label_mask in label_masks])
    pred_classes = np.array([pred_mask[0] for pred_mask in pred_masks])
    correct_class = true_classes[:, None] == pred_classes

    label_masks = np.load(label_npy_path)
    pred_masks = np.load(pred_npy_path)

    label_masks = label_masks.reshape((label_masks.shape[0], label_masks.shape[1]*label_masks.shape[2]))
    pred_masks = pred_masks.reshape((pred_masks.shape[0], pred_masks.shape[1]*pred_masks.shape[2]))
    intersection = np.dot(label_masks, pred_masks.T).clip(0)
    union = (label_masks.sum(axis=1)[:, np.newaxis] + pred_masks.sum(axis=1)[np.newaxis, :]) - intersection  # (area1 + area2) - intersection
    iou = intersection / (union + 1e-7)
    iou = iou * correct_class

    iou50 = iou >= 0.5
    correct_mask = correct_class & iou50


    correct_box_match = np.zeros_like(iou, dtype=bool)
    matches = np.array(np.nonzero(iou50)).T
    if matches.shape[0] != 0:
        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct_box_match[matches[:, 0], matches[:, 1]] = True

    gt_idx, pred_idx = np.nonzero(correct_box_match)
    match_dict = {int(gt):int(pred) for gt,pred in zip(gt_idx, pred_idx)}
    return match_dict


def process_files(label_path, pred_path, save_path, image_width=640, image_height=480, seg=False, with_conf=False,
                  with_att=False, threshold=0.5, att_num=0, label_npy_path=None, pred_npy_path=None):
    if seg:
        label_boxes, label_masks, labels_df = parse_masks(label_path, image_width, image_height, with_att=with_att, with_conf=with_conf)
        pred_boxes, pred_mask, pred_df = parse_masks(pred_path, image_width, image_height, with_att=with_att, with_conf=with_conf, pred=True)
        if label_npy_path is not None and pred_npy_path is not None:
            match_dict = match_masks_npy(label_masks, pred_mask, threshold, label_npy_path=label_npy_path, pred_npy_path=pred_npy_path)
        else:
            match_dict = match_masks(label_masks, pred_mask, threshold)
    else:
        label_boxes, labels_df = parse_boxes(label_path, image_width, image_height, with_att=with_att, with_conf=with_conf)
        pred_boxes, pred_df = parse_boxes(pred_path, image_width, image_height, with_att=with_att, with_conf=with_conf, pred=True)
        match_dict = match_boxes(label_boxes, pred_boxes, threshold)

    labels_df = labels_df.add_suffix('_labels')
    pred_df = pred_df.add_suffix('_pred')

    merge_df = pd.DataFrame(None, columns=labels_df.columns.tolist()+pred_df.columns.tolist())
    for i in range(len(label_boxes)):
        if i in match_dict:
            j = match_dict[i]
            record = labels_df.iloc[i].tolist()+pred_df.iloc[j].tolist()
        else:
            record = labels_df.iloc[i].tolist()+[None]*len(pred_df.columns)
        merge_df.loc[len(merge_df.index)] = record

    counts_all = []
    counts_div = []
    for i in range(att_num):
        merge_df[f'precision_att{i}'] = merge_df[f'att{i}_labels'] == merge_df[f'att{i}_pred']

        count_all = (merge_df[f'att{i}_labels'] == 1).sum()
        count_true = ((merge_df[f'att{i}_labels'] == 1) & (merge_df[f'att{i}_labels'] == merge_df[f'att{i}_pred'])).sum()
        count_div = count_true/count_all if count_all != 0 else np.nan
        counts_all.append(count_all)
        counts_div.append(count_div)

    precision_cols = merge_df.filter(like='precision_att').columns
    precision_means = merge_df[precision_cols].mean().tolist()
    mean_precision = np.nanmean(precision_means)

    mean_num = np.sum(counts_all)

    counts_div = [i for i in counts_div if not math.isnan(i)]
    mean_true = np.mean(counts_div) if len(counts_div)>0 else np.nan

    num_matched = len(match_dict)
    precision = num_matched / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = num_matched / len(label_boxes) if len(label_boxes) > 0 else 0.0

    col_xy = merge_df.pop('xy_labels')
    merge_df.insert(len(merge_df.columns), 'xy_labels', col_xy)
    col_xy = merge_df.pop('xy_pred')
    merge_df.insert(len(merge_df.columns), 'xy_pred', col_xy)
    merge_df.to_csv(save_path)
    return precision, recall, mean_precision, mean_true, mean_num, num_matched



def model_pred_compare(label_dir, pred_dir, save_dir=None, seg=False, with_conf=False, with_att=False, threshold=0.5,
                       image_width=640, image_height=480, att_num=0, labels_npy_dir=None, predict_npy_dir=None):
    box_num_sum = 0
    if save_dir is None:
        save_dir = label_dir + f'_pred_compare'
    columns = ['file_name', 'precision_box', 'recall_box', 'att_oa', 'att_oa_true', 'att_true_num',]
    os.makedirs(save_dir, exist_ok=True)
    pred_list = os.listdir(pred_dir)
    df = pd.DataFrame(None, columns=columns)
    for pred_file in tqdm(pred_list):
        label_path = os.path.join(label_dir, pred_file)
        save_path = os.path.join(save_dir, pred_file)
        result = [pred_file]
        pred_path = os.path.join(pred_dir, pred_file)
        if labels_npy_dir is not None and predict_npy_dir is not None:
            label_npy_path = os.path.join(labels_npy_dir, Path(pred_file).stem + '.npy')
            pred_npy_path = os.path.join(predict_npy_dir, Path(pred_file).stem + '.npy')
        if not os.path.exists(label_path) or not os.path.exists(pred_path):
            precision, recall, att_oa, att_oa_true, att_true_num, box_num = 0, 0, 0, 0, 0, 0
        else:
            precision, recall, att_oa, att_oa_true, att_true_num, box_num = process_files(label_path, pred_path, save_path, seg=seg,
                                                                                 image_width=image_width, image_height=image_height,
                                                                                 with_conf=with_conf, with_att=with_att,
                                                                                 threshold=threshold, att_num=att_num,
                                                                                 label_npy_path=label_npy_path,
                                                                                 pred_npy_path=pred_npy_path)
        result += [precision, recall, att_oa, att_oa_true, att_true_num]
        box_num_sum += box_num
        df.loc[len(df.index)] = result
    df.to_csv(save_dir+'.csv')
    print(f'totally {box_num_sum} box are counted')

def models_pred_compare(label_dir, preds_dir):
    pass

def img_merge(input_left_path, input_right_path, output_path):
    img_left = cv2.imread(input_left_path)
    img_right = cv2.imread(input_right_path)
    img_output = np.hstack((img_left, img_right))
    cv2.imwrite(output_path, img_output)

def imgs_merge(input_dir_left, input_dir_right, output_dir):
    img_list = os.listdir(input_dir_left)
    img_list.remove('labels')
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        input_img_left_path = os.path.join(input_dir_left, img_name)
        input_img_right_path = os.path.join(input_dir_right, img_name)
        output_img_path = os.path.join(output_dir, img_name)
        img_merge(input_img_left_path, input_img_right_path, output_img_path)

def cat_imgs(pre_path_list):
    pre_img_list = [cv2.imread(pre_path) for pre_path in pre_path_list]
    cat_img = np.concatenate(pre_img_list, axis=1)
    return cat_img

def cat_compare(pre_dir_list, show_dir):
    os.makedirs(show_dir, exist_ok=True)
    img_list = os.listdir(pre_dir_list[0])
    for img_name in tqdm(img_list):
        pre_path_list = [os.path.join(pre_dir, img_name) for pre_dir in pre_dir_list]
        cat_img = cat_imgs(pre_path_list)
        cv2.imwrite(os.path.join(show_dir, img_name), cat_img)

if __name__ == "__main__":
    pass
    # gt_dir = r'/localnvme/data/billboard/bd_data/data626_seg_c6/labels'
    # pred_dir = r'/localnvme/project/ultralytics/runs/segment/predict9/labels'
    # model_pred_compare(gt_dir, pred_dir, with_att=False, seg=True, att_num=0)


    # imgs_merge(r'/localnvme/data/billboard/ps_data/0516/images_split_pred/left',
    #            r'/localnvme/data/billboard/ps_data/0516/images_split_pred/right',
    #            r'/localnvme/data/billboard/ps_data/0516/images_split_pred/all')
    #
    # cat_compare([
    #     r'/localnvme/data/billboard/ps_data/0516/images_split_pred/all',
    #     r'/localnvme/data/billboard/ps_data/0516/images_pred',
    # ],
    # r'/localnvme/data/billboard/ps_data/0516/images_split_pred/compare',
    # )

    # imgs_merge(r'/localnvme/data/billboard/ps_data/0516/images_split_pred/left2',
    #            r'/localnvme/data/billboard/ps_data/0516/images_split_pred/right2',
    #            r'/localnvme/data/billboard/ps_data/0516/images_split_pred/all2')
    # fuse_mseg_c6_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6'
    # predict_dir = r'/localnvme/project/ultralytics/runs/msegment/val122/labels'

    # predict_dir = r'/localnvme/project/ultralytics/runs/msegment/val127/labels'
    # labels_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_update/labels_demo'
    # model_pred_compare(labels_dir, predict_dir, seg=True, with_conf=False, with_att=True, threshold=0.5, att_num=4,
    #                    image_width=1, image_height=1)

    # predict_dir = r'/localnvme/project/ultralytics/runs/msegment/val168/labels'
    # labels_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_update/labels'
    # labels_npy_dir = r'/localnvme/project/ultralytics/runs/msegment/val168/labels_npy'
    # predict_npy_dir = r'/localnvme/project/ultralytics/runs/msegment/val168/predicts_npy'
    # model_pred_compare(labels_dir, predict_dir, seg=True, with_conf=False, with_att=True, threshold=0.5, att_num=4,
    #                    image_width=1, image_height=1, labels_npy_dir=labels_npy_dir, predict_npy_dir=predict_npy_dir)
    fuse_mseg_c6_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618'
    labels_dir = os.path.join(fuse_mseg_c6_dir, 'labels')
    val_dir = os.path.join(r'/localnvme/project/ultralytics/runs/msegment', 'val207')
    predict_dir = os.path.join(val_dir, 'labels')
    labels_npy_dir = os.path.join(val_dir, 'labels_npy')
    predict_npy_dir = os.path.join(val_dir, 'predicts_npy')
    save_dir = predict_dir + 'val207_compare'

    model_pred_compare(labels_dir, predict_dir, save_dir=save_dir, seg=True, with_conf=False, with_att=True,
                       threshold=0.5, att_num=4, image_width=1, image_height=1, labels_npy_dir=labels_npy_dir,
                       predict_npy_dir=predict_npy_dir)
