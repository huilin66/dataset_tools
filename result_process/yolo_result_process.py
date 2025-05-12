import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from config import *

def compute_iou_box(box1, box2):
    xc1,xc2 = box1[0], box2[0]
    yc1,yc2 = box1[1], box2[1]
    w1,w2 = box1[2], box2[2]
    h1,h2 = box1[3], box2[3]
    x1_tp = xc1-w1*0.5
    x2_tp = xc2-w2*0.5
    y1_tp = yc1-h1*0.5
    y2_tp = yc2-w2*0.5

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
        columns = ['att_%d'%i for i in range(len(atts[0]))]
        df_att = pd.DataFrame(atts, columns=columns)
        df = pd.concat([df,df_att], axis=1)
    boxes_abs = box2box_abs(boxes, image_width, image_height)
    return boxes_abs, df


def parse_masks(file_path, image_width, image_height, pred=False, with_att=False, with_conf=False):
    boxes = []
    masks = []
    df = pd.DataFrame(None, columns=['class', 'xy', 'conf'])
    atts = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
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
            atts.append(att)

            if with_conf:
                conf = float(parts[-1])
                mask = list(map(float, parts[coord_idx:-1]))
            else:
                conf = 1
                mask = list(map(float, parts[coord_idx:]))
            df.loc[len(df)] = [cls, mask, conf]
            polygon = np.array(mask).reshape(-1, 2)
            polygon[:, 0] = polygon[:, 0] * image_width
            polygon[:, 1] = polygon[:, 1] * image_height
            box = poly2xywh(polygon)
            boxes.append([cls]+box)
    if with_att:
        columns = ['att_%d' % i for i in range(len(atts[0]))]
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
    for i in range(0, len(masks), 2):
        abs_masks[i] = masks[i]*W
        abs_masks[i+1] = masks[i+1] * H
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
    cost_matrix = np.full((num_labels, num_preds), 100)
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
        if cost_matrix[r][c] != np.inf:
            match_dict[r] = c
    return match_dict


def process_files(label_path, pred_path, save_path, image_width=640, image_height=480, seg=False, with_conf=False,
                  with_att=False, threshold=0.5):
    if seg:
        label_boxes, label_masks, labels_df = parse_masks(label_path, image_width, image_height)
        pred_boxes, pred_mask, pred_df = parse_masks(pred_path, image_width, image_height, pred=True)
    else:
        label_boxes, labels_df = parse_boxes(label_path, image_width, image_height)
        pred_boxes, pred_df = parse_boxes(pred_path, image_width, image_height, pred=True)


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
    # for i in range(9):
    # for i in [1, 4, 5, 6, 7]:
    for i in [0, 2, 3]:
        merge_df[f'precision_att{i+1}'] = merge_df[f'att{i+1}_labels'] == merge_df[f'att{i+1}_pred']
        # merge_df[f'true_att{i+1}'] = (merge_df[f'att{i+1}_labels'] == 1) & (merge_df[f'att{i+1}_labels'] == merge_df[f'att{i+1}_pred'])

        count_all = (merge_df[f'att{i+1}_labels'] == 1).sum()
        count_true = ((merge_df[f'att{i+1}_labels'] == 1) & (merge_df[f'att{i+1}_labels'] == merge_df[f'att{i+1}_pred'])).sum()
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
    merge_df.to_csv(save_path)
    return precision, recall, mean_precision, mean_true, mean_num



def model_pred_compare(label_dir, pred_dir, save_dir=None, seg=False, with_conf=False, with_att=False, threshold=0.5):
    if save_dir is None:
        save_dir = label_dir + f'_{os.path.dirname(pred_dir)}'
    columns = ['file_name', 'precision_box', 'recall_box', 'att_oa', 'att_oa_true', 'att_true_num',]
    os.makedirs(save_dir, exist_ok=True)
    pred_list = os.listdir(pred_dir)
    df = pd.DataFrame(None, columns=columns)
    for pred_file in tqdm(pred_list):
        label_path = os.path.join(label_dir, pred_file)
        save_path = os.path.join(save_dir, pred_file)
        result = [pred_file]
        pred_path = os.path.join(pred_dir, pred_file)
        if not os.path.exists(label_path) or not os.path.exists(pred_path):
            precision, recall, att_oa, att_oa_true, att_true_num = 0, 0, 0, 0, 0
        else:
            precision, recall, att_oa, att_oa_true, att_true_num = process_files(label_path, pred_path, save_path, seg=seg,
                                                                                 with_conf=with_conf, with_att=with_att,
                                                                                 threshold=threshold)
        result += [precision, recall, att_oa, att_oa_true, att_true_num]
        df.loc[len(df.index)] = result
    df.to_csv(save_path)

def models_pred_compare(label_dir, preds_dir):
    pass

if __name__ == "__main__":
    pass
    model_pred_compare(ISDS_DATA389C6_LABEL, ISDS_DATA389C6_PREDCIT, with_att=True, seg=True)


