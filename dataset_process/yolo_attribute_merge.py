import os
import pandas as pd
from tqdm import tqdm


id_merge_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
}


def yolo_att_merge(gt_dir, merge_dict):
    gt_list = os.listdir(gt_dir)
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        df = pd.read_csv(gt_path, names=['catid', 'x1', 'x2', 'y1', 'y2'], header=None, index_col=None, sep=' ')
        for idx, row in df.iterrows():
            src_id = row['catid']
            dst_id = merge_dict[src_id]
            df.loc[idx, 'catid'] = dst_id
        df.to_csv(gt_path, header=None, index=None, sep=' ')
