import os
from tqdm import tqdm
import pandas as pd

right_order = [
    'Automatic Sprinkler',
    'Fire Detectors',
    'Fire Alarm Bell',
    'Fire Alarm_break grass',
    'Fire Hose Reel',
    'Exit',
]

# wrong_name : right_name
wrong_order_map = {
    'Automatic Sprinkler': 'Exit',
    'Fire Detectors': 'Automatic Sprinkler',
    'Fire Alarm Bell': 'Fire Hose Reel',
    'Fire Alarm_break grass':'Fire Alarm_break grass',
    'Fire Hose Reel': 'Fire Alarm Bell',
    'Exit': 'Fire Detectors',
}

def get_id2cat(cats):
    return dict(zip(range(len(cats)), cats))
def get_cat2id(cats):
    return dict(zip(cats, range(len(cats))))

def order_correction(gt_dir):
    id2cat = get_id2cat(right_order)
    cat2id = get_cat2id(right_order)
    gt_list = os.listdir(gt_dir)
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        df = pd.read_csv(gt_path, names=['catid', 'x1', 'x2', 'y1', 'y2'], header=None, index_col=None, sep=' ')
        for idx, row in df.iterrows():
            src_id = row['catid']
            src_name = id2cat[src_id]
            dst_name = wrong_order_map[src_name]
            dst_id = cat2id[dst_name]
            df.loc[idx, 'catid'] = dst_id
        df.to_csv(gt_path, header=None, index=None, sep=' ')

if __name__ == '__main__':
    pass
    gt_dir = r'E:\data\0318_fireservice\data0325\labels_correction'
    order_correction(gt_dir)
    
    # print(get_id2cat(right_order))
    # print(get_cat2id(right_order))