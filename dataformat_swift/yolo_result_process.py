import os
import math
import numpy as np
import pandas as pd
from datasets import tqdm
from scipy.optimize import linear_sum_assignment


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def parse_boxes(file_path, pred=False):
    boxes = []
    df = pd.DataFrame(None, columns=['class', 'x1', 'y1', 'x2', 'y2', 'att1', 'att2','att3','att4','att5','att6', 'att7','att8','att9','att10'])
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            cls = int(parts[0])
            x = float(parts[-4])
            y = float(parts[-3])
            h = float(parts[-2])
            w = float(parts[-1])
            boxes.append((cls, x, y, h, w))
            if pred:
                atts = list(map(float, parts[2:-4]))
                atts = [1 if x > 0.0 else 0 for x in atts]
            else:
                atts = list(map(int, parts[2:-4]))
            df.loc[len(df)] = [cls, x, y, h, w] + atts
    return boxes, df


def convert_to_abs(boxes, W, H):
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


def compute_cost_matrix(label_boxes, pred_boxes, threshold=0.5):
    n_labels = len(label_boxes)
    n_preds = len(pred_boxes)
    cost_matrix = np.full((n_labels, n_preds), np.inf)  # 初始化为 inf

    for i, label_box in enumerate(label_boxes):
        for j, pred_box in enumerate(pred_boxes):
            if label_box[0] == pred_box[0]:  # 仅同类匹配
                iou = compute_iou(label_box, pred_box)
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
            iou = compute_iou((x1_l, y1_l, x2_l, y2_l), (x1_p, y1_p, x2_p, y2_p))
            if iou >= threshold:
                cost_matrix[i][j] = -iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match_dict = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r][c] != np.inf:
            match_dict[r] = c
    return match_dict


def process_files(label_path, pred_path, save_path, image_width=640, image_height=480, threshold=0.5):
    label_boxes, labels_df = parse_boxes(label_path)
    if not os.path.exists(pred_path):
        return 0, 0, 0, 0, 0
    pred_boxes, pred_df = parse_boxes(pred_path, pred=True)

    label_abs = convert_to_abs(label_boxes, image_width, image_height)
    pred_abs = convert_to_abs(pred_boxes, image_width, image_height)

    match_dict = match_boxes(label_abs, pred_abs, threshold)

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


def process_dirs(label_dir, pred_dirs, save_dir, threshold=0.5):
    pass
    os.makedirs(save_dir, exist_ok=True)
    pred_list = os.listdir(pred_dirs[0])
    df = pd.DataFrame(None, columns=['file_name',
                                     'precision_box_v8', 'recall_box_v8', 'att_oa_v8', 'att_oa_true_v8', 'att_true_num_v8',
                                     'precision_box_v9', 'recall_box_v9', 'att_oa_v9', 'att_oa_true_v9', 'att_true_num_v9',
                                     'precision_box_v10', 'recall_box_v10', 'att_oa_v10', 'att_oa_true_v10', 'att_true_num_v10',
                                     'precision_box_vma', 'recall_box_vma', 'att_oa_vma', 'att_oa_true_vma', 'att_true_num_vma',
                                     ])
    for pred_file in tqdm(pred_list):
        label_path = os.path.join(label_dir, pred_file)
        save_path = os.path.join(save_dir, pred_file)
        result = [pred_file]
        for pred_dir in pred_dirs:
            pred_path = os.path.join(pred_dir, pred_file)
            precision, recall, att_oa, att_oa_true, att_true_num = process_files(label_path, pred_path, save_path, threshold=threshold)
            result += [precision, recall, att_oa, att_oa_true, att_true_num]
        df.loc[len(df.index)] = result
    df.to_csv(save_dir+'_filter2.csv')


def process_dir(label_dir, pred_dir, save_dir, threshold=0.5):
    pass
    os.makedirs(save_dir, exist_ok=True)
    pred_list = os.listdir(pred_dir)
    df = pd.DataFrame(None, columns=['file_name', 'precision_box', 'recall_box', 'att_oa', 'att_oa_true', 'att_true_num'])
    for pred_file in tqdm(pred_list):
        label_path = os.path.join(label_dir, pred_file)
        pred_path = os.path.join(pred_dir, pred_file)
        save_path = os.path.join(save_dir, pred_file)
        precision, recall, att_oa, att_oa_true, att_true_num = process_files(label_path, pred_path, save_path, threshold=threshold)
        df.loc[len(df.index)] = [pred_file, precision, recall, att_oa, att_oa_true, att_true_num]
    df.to_csv(save_dir+'.csv')

# 示例用法
if __name__ == "__main__":
    pass
    # label_dir = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels"
    # pred_dir = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4\labels"
    # save_dir = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4_infer"
    # process_dir(label_dir, pred_dir, save_dir)

    label_dir = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels"
    pred_dirs = [
        r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_yolo8x2\labels",
        r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_yolo9e2\labels",
        r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_yolo10x2\labels",
        r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4\labels",
    ]
    save_dir = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_compare"
    process_dirs(label_dir, pred_dirs, save_dir)


    # label_file = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels\FLIR1312.txt"
    # pred_file = r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict_\predict_mayolox4\labels\FLIR1312.txt"
    # image_width = 640  # 替换为实际图片宽度
    # image_height = 480  # 替换为实际图片高度
    # threshold = 0.5
    #
    # process_files(label_file, pred_file, None, image_width, image_height, threshold)

