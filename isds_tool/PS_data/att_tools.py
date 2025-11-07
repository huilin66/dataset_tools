import os
import shutil
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from yolo_tools import get_yolo_label_df, get_attributes

def list_remove_index(input_list, remove_index):
    output_list = [item for i, item in enumerate(input_list) if i not in remove_index]
    return output_list

def extract_single_risk_keep_len(input_dir, output_dir, risk):
    save_dir = os.path.join(output_dir, risk, 'labels')
    os.makedirs(save_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list, desc=risk):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(save_dir, label_name)
        with open(input_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                if risk == 'd':
                    parts[3], parts[4], parts[5] = '0', '0', '0'
                elif risk == 'b':
                    parts[2], parts[4], parts[5] = '0', '0', '0'
                elif risk == 'a':
                    parts[2], parts[3], parts[5] = '0', '0', '0'
                elif risk == 'c':
                    parts[2], parts[3], parts[4] = '0', '0', '0'
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)


def extract_single_risk_keep_single(input_dir, output_dir, risk):
    save_dir = os.path.join(output_dir, risk, 'labels')
    os.makedirs(save_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list, desc=risk):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(save_dir, label_name)
        with open(input_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                parts[1] = '1'
                if risk == 'd':
                    parts = list_remove_index(parts, [3, 4, 5])
                elif risk == 'b':
                    parts = list_remove_index(parts, [2, 4, 5])
                elif risk == 'a':
                    parts = list_remove_index(parts, [2, 3, 5])
                elif risk == 'c':
                    parts = list_remove_index(parts, [2, 3, 4])
                else:
                    ValueError(f"{risk} risk must be 'd' or 'b' or 'a' or 'c'")
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)




def risk_refine_single(input_gt_dir, output_gt_dir, ref_dir,  risk='b'):
    os.makedirs(output_gt_dir, exist_ok=True)
    gt_list = os.listdir(input_gt_dir)
    ref_list = [Path(file_name).stem for file_name in os.listdir(ref_dir)]

    # c 4 d b a c
    if risk == 'd':
        risk_index = 2
    elif risk == 'b':
        risk_index = 3
    elif risk == 'a':
        risk_index = 4
    elif risk == 'c':
        risk_index = 5
    else:
        ValueError(f"{risk} risk must be 'd' or 'b' or 'a' or 'c'")

    diff_count = 0
    for label_name in tqdm(gt_list):
        input_gt_path = os.path.join(input_gt_dir, label_name)
        output_gt_path = os.path.join(output_gt_dir, label_name)
        with open(input_gt_path, 'r') as fi:
            lines = fi.readlines()
            new_lines = []
            for id_line, line in enumerate(lines):
                parts = line.strip().split(' ')

                obj_name = Path(label_name).stem + f'_{id_line}'
                if obj_name in ref_list:
                    if parts[risk_index] != '0':
                        diff_count += 1
                    parts[risk_index] = '0'
                new_line = ' '.join(parts) +'\n'
                new_lines.append(new_line)
        with open(output_gt_path, 'w') as fo:
            fo.writelines(new_lines)
    print(f'change {diff_count}, all {len(ref_list)}')


def risk_change_line(parts, src_risk, dst_risk):
    # c 4 d b a c
    change = False
    if src_risk == 'a-h' and dst_risk == 'b-h':
        if parts[4] == '2':
            parts[4] = '0'
            parts[3] = '2'
            change = True
    return parts, change

def risk_change(input_gt_dir, output_gt_dir, src_risk='a-h', dst_risk='b-h'):
    os.makedirs(output_gt_dir, exist_ok=True)
    gt_list = os.listdir(input_gt_dir)

    change_count = 0
    for label_name in tqdm(gt_list):
        input_gt_path = os.path.join(input_gt_dir, label_name)
        output_gt_path = os.path.join(output_gt_dir, label_name)
        with open(input_gt_path, 'r') as fi:
            lines = fi.readlines()
            new_lines = []
            for id_line, line in enumerate(lines):
                parts = line.strip().split(' ')
                parts, change = risk_change_line(parts, src_risk, dst_risk)
                if change:
                    change_count += 1
                new_line = ' '.join(parts) +'\n'
                new_lines.append(new_line)
        with open(output_gt_path, 'w') as fo:
            fo.writelines(new_lines)
    print(f'change {change_count}')


def get_img_list(input_csv_path):
    df = pd.read_csv(input_csv_path, index_col=None, names=['file_name'])
    img_list = df['file_name'].to_list()
    img_list = [os.path.basename(file_path) for file_path in img_list]
    return img_list

def remove_repeated(input_dir, ref_dir):
    common_list = find_common_list(input_dir, ref_dir)
    for common_name in tqdm(common_list):
        input_path = os.path.join(input_dir, common_name)
        os.remove(input_path)

def find_common_list(input_list1, input_list2):
    if isinstance(input_list1, list):
        input_list1 = input_list1
    elif os.path.isdir(input_list1):
        input_list1 = os.listdir(input_list1)
    else:
        ValueError(input_list1, 'error!')
    if isinstance(input_list2, list):
        input_list2 = input_list2
    elif os.path.isdir(input_list2):
        input_list2 = os.listdir(input_list2)
    else:
        ValueError(input_list2, 'error!')

    common_list = []
    for img_path in input_list1:
        if img_path in input_list2:
            common_list.append(img_path)
    return common_list

def data_check(input_dir):
    all_path = os.path.join(input_dir, 'all.txt')
    train_path = os.path.join(input_dir, 'train_80p_ref.txt')
    val_path = os.path.join(input_dir, 'val_80p_ref.txt')
    all_test_path = os.path.join(input_dir, 'all_test.txt')
    train_test_path = os.path.join(input_dir, 'train_test.txt')
    val_test_path = os.path.join(input_dir, 'val_test.txt')

    img_list_all = get_img_list(all_path)
    img_list_train = get_img_list(train_path)
    img_list_val = get_img_list(val_path)
    img_list_all_test = get_img_list(all_test_path)
    img_list_train_test = get_img_list(train_test_path)
    img_list_val_test = get_img_list(val_test_path)

    print(f'all data {len(img_list_all)}, train data {len(img_list_train)}, val data {len(img_list_val)}')
    common_list = find_common_list(img_list_train, img_list_val)
    print(f'find {len(common_list)}, between train & val')
    common_list = find_common_list(img_list_all, img_list_train)
    print(f'find {len(common_list)}, between train & all')
    common_list = find_common_list(img_list_all, img_list_val)
    print(f'find {len(common_list)}, between val & all')
    print()

    print(f'all test data {len(img_list_all_test)}, train data {len(img_list_train_test)}, val data {len(img_list_val_test)}')
    common_list = find_common_list(img_list_train_test, img_list_val_test)
    print(f'find {len(common_list)}, between test train & val')
    common_list = find_common_list(img_list_all_test, img_list_train_test)
    print(f'find {len(common_list)}, between test train & all')
    common_list = find_common_list(img_list_all_test, img_list_val_test)
    print(f'find {len(common_list)}, between test val & all')
    print()

    common_list = find_common_list(img_list_all_test, img_list_all)
    print(f'find {len(common_list)}, between test all & all')

def copy_files(input_dir, output_dir):
    file_list = os.listdir(input_dir)
    print(f'find {len(file_list)}, copying...')
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm(file_list, desc=f'copy {os.path.basename(input_dir)} -> {os.path.basename(output_dir)}'):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.copyfile(input_path, output_path)
    print(f'copy {len(file_list)} to {os.path.basename(output_dir)}, get {len(os.listdir(output_dir))}')

def copy_files_list(input_dir_list, output_dir):
    if isinstance(input_dir_list, list):
        for input_dir in input_dir_list:
            copy_files(input_dir, output_dir)
    elif isinstance(input_dir_list, str):
        if os.path.isdir(input_dir_list):
            copy_files(input_dir_list, output_dir)
        else:
            ValueError(input_dir_list, 'error!')
    else:
        ValueError(input_dir_list, 'error!')


def box_iou(box1, box2, eps=1e-7):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / (union + eps)

def match_and_merge(df_pred, df_gt, iou_thr=0.5, att_list=None):
    df_gt = df_gt.reset_index(drop=True)
    matched_gt_idx = set()
    merged_rows = []

    show_columns = ['id', "category", "x", "y", "w", "h", "x1", "y1", "x2", "y2"]
    if att_list is not None:
       show_columns += att_list + ['defect']
    for i, pred_row in df_pred.iterrows():
        pred_box = [pred_row.x1, pred_row.y1, pred_row.x2, pred_row.y2]
        ious = [box_iou(pred_box, [gt.x1, gt.y1, gt.x2, gt.y2]) for _, gt in df_gt.iterrows()]
        if ious:
            max_iou = max(ious)
            max_idx = np.argmax(ious)
        else:
            max_iou = 0
            max_idx = None

        if max_iou > iou_thr and max_idx not in matched_gt_idx:
            gt_row = df_gt.iloc[max_idx]
            matched_gt_idx.add(max_idx)
            merged_rows.append({
                **{f"pred_{col}": pred_row[col] for col in show_columns},
                **{f"gt_{col}": gt_row[col] for col in show_columns},
                "iou": max_iou
            })
        else:
            merged_rows.append({
                **{f"pred_{col}": pred_row[col] for col in show_columns},
                **{f"gt_{col}": None for col in show_columns},
                "iou": None
            })

    # 把没被匹配的 GT 也补上
    for j, gt_row in df_gt.iterrows():
        if j not in matched_gt_idx:
            merged_rows.append({
                **{f"pred_{col}": None for col in show_columns},
                **{f"gt_{col}": gt_row[col] for col in show_columns},
                "iou": None
            })

    return pd.DataFrame(merged_rows)

class HDFManager:
    def __init__(self, path, mode='r'):
        self.path = path
        self.store = pd.HDFStore(path, mode=mode)
    def get(self, key):
        return self.store['/'+key]
    def close(self):
        self.store.close()

def load_all_label(label_dir, attributes):
    label_all_path = label_dir+'_df_all.h5'
    if os.path.exists(label_all_path):
        print(f'load {label_all_path}...')
        label_df_dict = HDFManager(label_all_path)
        # with pd.HDFStore(label_all_path, model='r') as store:
        #     for k in tqdm(store.keys()):
        #         label_df_dict[k.lstrip('/')] = store[k]
        print('finish!\n')
    else:
        label_list = os.listdir(label_dir)

        label_df_dict = {}
        for label_name in tqdm(label_list, desc='load label df'):
            file_stem = Path(label_name).stem
            input_label_path = os.path.join(label_dir, label_name)
            df_label = get_yolo_label_df(input_label_path, mdet=True, attributes=attributes, with_object_id=True)
            label_df_dict[file_stem] = df_label
        print(f'save {label_all_path}...')
        with pd.HDFStore(label_all_path, model='w') as store:
            for k, df in label_df_dict.items():
                store.put(k, df, format='table')
        label_df_dict = HDFManager(label_all_path)
        print('finish!\n')
    return label_df_dict



def pred_check_iou(label_dir, pred_dir, pred_obj_dir, attributes=None, with_conf=True, conf_threshold=0.3, iou_thr=0.3, defect_conf_threshold=None):
    attributes = get_attributes(attributes)
    label_df_dict = load_all_label(label_dir, attributes)

    match_obj_list = []
    pred_obj_list = os.listdir(pred_obj_dir)
    for object_name in tqdm(pred_obj_list, desc='load pred_obj'):
        object_stem = Path(object_name).stem
        file_stem, object_id = object_stem.rsplit('_', 1)
        pred_path = os.path.join(pred_dir, file_stem+'.txt')
        df_pred = get_yolo_label_df(pred_path, mdet=True, attributes=attributes, with_object_id=True,
                                   with_conf=with_conf, conf_threshold=conf_threshold,
                                    defect_conf_threshold=defect_conf_threshold)
        df_pred_obj = df_pred[df_pred['id']==int(object_id)]
        if len(df_pred_obj) != 1:
            print(f'{object_name} match error')

        df_match = match_and_merge(df_pred_obj, label_df_dict.get(file_stem), iou_thr=iou_thr, att_list=attributes)
        df_match = df_match[~df_match['pred_id'].isna()]
        if len(df_match) == 1:
            if not df_match['gt_id'].isna().all():
                match_obj_list.append(object_name)
        else:
            print(f'{object_name} match error')
    print(f'find {len(match_obj_list)} matched records in {len(pred_obj_list)}')
    save_path = pred_obj_dir+'_iou_check.csv'
    df = pd.DataFrame({'path': match_obj_list})
    df.to_csv(save_path, header=False, index=False)

def remove_by_list(ref_list, input_dir, output_dir, ref_keep=True):
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(ref_list, str):
        if os.path.isfile(ref_list):
            ref_df = pd.read_csv(ref_list, names=['path'])
            ref_list = ref_df['path'].to_list()
        elif os.path.isdir(ref_list):
            ref_list = os.listdir(ref_list)
        else:
            ValueError(f'{ref_list} error')
    elif isinstance(ref_list, list):
        pass
    else:
        ValueError(f'{ref_list} error')
    os.makedirs(output_dir, exist_ok=True)

    input_list = os.listdir(input_dir)
    for input_name in tqdm(input_list, desc='remove by list'):
        if (ref_keep and input_name in ref_list) or (not ref_keep and input_name not in input_list):
            input_path = os.path.join(input_dir, input_name)
            output_path = os.path.join(output_dir, input_name)
            if os.path.isfile(input_path):
                shutil.copy(input_path, output_path)
            else:
                print(f'{input_name} not exist')
    print(f'remove {len(ref_list)} records, {len(input_list)} -> {len(os.listdir(input_dir))}')

def pred2label(pred_dir, pred_obj_dir, output_pred_dir, with_conf=True):
    os.makedirs(output_pred_dir, exist_ok=True)
    pred_obj_list = os.listdir(pred_obj_dir)
    label_pred_dict = {}
    for object_name in tqdm(pred_obj_list, desc='load pred_obj'):
        object_stem = Path(object_name).stem
        file_stem, object_id = object_stem.rsplit('_', 1)
        object_id = int(object_id)
        file_name = file_stem+'.txt'
        if file_name in label_pred_dict:
            label_pred_dict[file_name].append(object_id)
        else:
            label_pred_dict[file_name] = [object_id]

    print(f'load {len(label_pred_dict)}/{len(pred_obj_list)} records,')

    change_image_count, change_obj_count = 0, 0
    pred_list = os.listdir(pred_dir)
    for pred_name in tqdm(pred_list, desc='load pred_list'):
        if pred_name in label_pred_dict:
            change_image_count += 1
            input_pred_path = os.path.join(pred_dir, pred_name)
            output_pred_path = os.path.join(output_pred_dir, pred_name)
            with open(input_pred_path, 'r') as fr:
                lines = fr.readlines()
                new_lines = []
                for line_id, line in enumerate(lines):
                    if line_id in label_pred_dict[pred_name]:
                        change_obj_count += 1
                        parts = line.strip().split(' ')
                        assert parts[1] == '4', f"{pred_name} error"
                        parts[2], parts[3], parts[4], parts[5] = '0', '0', '0', '0'
                        if with_conf:
                            parts = parts[:-1]
                        new_line = ' '.join(parts)+'\n'
                        new_lines.append(new_line)
                with open(output_pred_path, 'w') as fw:
                    fw.writelines(new_lines)

    print(f'change {change_image_count}/{change_obj_count} records,')


def get_risk_by_ref(risk_objects_dict, risks, pred_name, line_id):
    object_name = f'{Path(pred_name).stem}_{line_id}'
    results = ['0', '0', '0', '0']
    for id, risk in enumerate(risks):
        ph_lb_str = f'risk_{risk}_pred_high_label_background'
        pm_lb_str = f'risk_{risk}_pred_medium_label_background'
        if object_name in risk_objects_dict[ph_lb_str]:
            results[id] = '2'
        elif object_name in risk_objects_dict[pm_lb_str]:
            results[id] = '1'
    return results


def get_risk_objects(ref_dir, risks):
    risk_objects_dict = {}
    pred_high_label_background_dir = os.path.join(ref_dir, 'pred_high_label_background')
    pred_medium_label_background_dir = os.path.join(ref_dir, 'pred_medium_label_background')
    for risk in risks:
        ph_lb_r = os.path.join(pred_high_label_background_dir, f'risk_{risk}_pred_high_label_background')
        pm_lb_r = os.path.join(pred_medium_label_background_dir, f'risk_{risk}_pred_medium_label_background')
        risk_objects_dict[f'risk_{risk}_pred_high_label_background'] = [Path(file_name).stem for file_name in
                                                                        os.listdir(ph_lb_r)]
        risk_objects_dict[f'risk_{risk}_pred_medium_label_background'] = [Path(file_name).stem for file_name in
                                                                        os.listdir(pm_lb_r)]
    return risk_objects_dict

def pred2label_ref(pred_dir, pred_obj_dir, output_pred_dir, ref_dir, attributes=None, with_conf=True):
    attributes = get_attributes(attributes)
    risk_objects_dict = get_risk_objects(ref_dir, attributes)

    shutil.rmtree(output_pred_dir) if os.path.exists(output_pred_dir) else None
    os.makedirs(output_pred_dir, exist_ok=True)
    pred_obj_list = os.listdir(pred_obj_dir)
    label_pred_dict = {}
    for object_name in tqdm(pred_obj_list, desc='load pred_obj'):
        object_stem = Path(object_name).stem
        file_stem, object_id = object_stem.rsplit('_', 1)
        object_id = int(object_id)
        file_name = file_stem+'.txt'
        if file_name in label_pred_dict:
            label_pred_dict[file_name].append(object_id)
        else:
            label_pred_dict[file_name] = [object_id]
    print(f'load {len(label_pred_dict)}/{len(pred_obj_list)} records,')

    change_image_count, change_obj_count = 0, 0
    pred_list = os.listdir(pred_dir)
    for pred_name in tqdm(pred_list, desc='load pred_list'):
        if pred_name in label_pred_dict:
            change_image_count += 1
            input_pred_path = os.path.join(pred_dir, pred_name)
            output_pred_path = os.path.join(output_pred_dir, pred_name)
            with open(input_pred_path, 'r') as fr:
                lines = fr.readlines()
                new_lines = []
                for line_id, line in enumerate(lines):
                    if line_id in label_pred_dict[pred_name]:
                        change_obj_count += 1
                        parts = line.strip().split(' ')
                        assert parts[1] == '4', f"{pred_name} error"
                        risks = get_risk_by_ref(risk_objects_dict, attributes, pred_name, line_id)
                        parts[2], parts[3], parts[4], parts[5] = risks
                        if with_conf:
                            parts = parts[:-1]
                        new_line = ' '.join(parts)+'\n'
                        new_lines.append(new_line)
                with open(output_pred_path, 'w') as fw:
                    fw.writelines(new_lines)

    print(f'change {change_image_count}/{change_obj_count} records,')


def labels_merge(input_dir1, input_dir2, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_dir_list1 = os.listdir(input_dir1)
    input_dir_list2 = os.listdir(input_dir2)
    for label_name in tqdm(input_dir_list1, desc='merge'):
        input_path1 = os.path.join(input_dir1, label_name)
        input_path2 = os.path.join(input_dir2, label_name)
        output_path = os.path.join(output_dir, label_name)
        if label_name not in input_dir_list2:
            shutil.copy(input_path1, output_path)
        else:
            with open(input_path1, 'r') as fr1, open(input_path2, 'r') as fr2:
                lines1 = fr1.readlines()
                lines2 = fr2.readlines()
                new_lines = lines1 + lines2
                with open(output_path, 'w') as fw:
                    fw.writelines(new_lines)

def get_cats(class_file):
    df = pd.read_csv(class_file, header=None, index_col=None, names=['category'])
    cats = df['category'].to_list()
    return cats

def get_category_by_ref(ref_dir, cats, label_name, line_id):
    obj_name = f'{Path(label_name).stem}_{line_id}'
    cat_result=None
    for id, cat in enumerate(cats):
        cat_dir = os.path.join(ref_dir, cat)
        if os.path.exists(cat_dir) and os.path.exists(os.path.join(cat_dir, obj_name)):
            cat_result = id
    no_dir = os.path.join(ref_dir, 'no')
    if os.path.exists(no_dir) and os.path.exists(os.path.join(no_dir, obj_name)):
        cat_result = -1
    if cat_result is None:
        ValueError(f'cannot find {label_name} in {ref_dir}')
    else:
        return str(cat_result)

def category_update_by_ref(input_dir, output_dir, ref_dir, class_file):
    cats = get_cats(class_file)

    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)
    label_obj_dict = {}
    obj_count = 0
    for cat in cats+['no']:
        cat_dir = os.path.join(ref_dir, cat)
        if not os.path.exists(cat_dir):
            print(f'{cat} not exists')
            continue
        obj_list = os.listdir(cat_dir)
        obj_count += len(obj_list)
        for object_name in tqdm(obj_list, desc=f'load {cat}'):
            object_stem = Path(object_name).stem
            file_stem, object_id = object_stem.rsplit('_', 1)
            object_id = int(object_id)
            file_name = file_stem+'.txt'
            if file_name in label_obj_dict:
                label_obj_dict[file_name].append(object_id)
            else:
                label_obj_dict[file_name] = [object_id]
    print(f'load {len(label_obj_dict)}/{obj_count} records,')


    change_image_count, change_obj_count = 0, 0
    input_list = os.listdir(input_dir)
    for label_name in tqdm(input_list, desc='update label'):
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)

        if label_name not in label_obj_dict:
            shutil.copy(input_label_path, output_label_path)
        else:
            change_image_count += 1
            with open(input_label_path, 'r') as fr:
                lines = fr.readlines()
                new_lines = []
                for line_id, line in enumerate(lines):
                    if line_id in label_obj_dict[label_name]:
                        change_obj_count += 1
                        parts = line.strip().split(' ')
                        cat = get_category_by_ref(ref_dir, cats, label_name, line_id)
                        if cat is None or cat == 'None':
                            ValueError(f'cannot find {label_name} in {ref_dir}')
                        elif cat != '-1':
                            parts[0] = cat
                        else:
                            continue
                        new_line = ' '.join(parts) + '\n'
                    else:
                        new_line = line
                    new_lines.append(new_line)
                with open(output_label_path, 'w') as fw:
                    fw.writelines(new_lines)

    print(f'change {change_image_count}/{change_obj_count} records, total {len(input_list)} records,')
if __name__ == '__main__':
    pass
    # pred_check_iou(
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/labels',
    #     r'/localnvme/project/ultralytics/runs/msegment/val694/labels',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_without_risk',
    #     # r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_with_risk',
    #     attributes=r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/attribute.yaml',
    # )

    # remove_by_list(
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_without_risk_iou_check.csv',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_without_risk',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_without_risk_iou_check',
    # )
    # remove_by_list(
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_with_risk_iou_check.csv',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_with_risk',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/all_with_risk_iou_check',
    # )

    # pred2label(
    #     r'/localnvme/project/ultralytics/runs/msegment/val694/labels',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/merge_1105/all',
    #     r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/merge_1105_labels',
    #     attributes=r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/attribute.yaml',
    # )
    copy_files_list(r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background/merge_1106/all_with_risk',
                    None)