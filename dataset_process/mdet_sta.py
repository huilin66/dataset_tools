import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def mdet_sta(input_dir, output_txt, attribute_len=14):
    dfs = []
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, names=['category', 'attribute_len']+['att_%i'%i for i in range(attribute_len)]+['x1', 'y1', 'x2', 'y2'], header=None, index_col=None, sep=' ')
        dfs.append(df)
    dfs = pd.concat(dfs)

    # attribute_num = [0]*(attribute_len+1)
    str_att_num, str_att_rate = 'attribute numbers:', 'attribute rates  :'
    for i in range(attribute_len):
        att_num = dfs['att_%i'%i].sum()
        att_rate = dfs['att_%i'%i].mean()
        str_att_num += '%10d'%(att_num)
        str_att_rate += '%10.6f'%(att_rate)
    print(str_att_num)
    print(str_att_rate)

def mdet_sta_val(input_dir, output_txt, input_txt, attribute_len=14):
    val_list = pd.read_csv(input_txt, header=None, index_col=None)[0].tolist()
    val_list = [Path(file_name).stem for file_name in val_list]

    dfs = []
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        if Path(file_name).stem not in val_list:
            continue
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, names=['category', 'attribute_len']+['att_%i'%i for i in range(attribute_len)]+['x1', 'y1', 'x2', 'y2'], header=None, index_col=None, sep=' ')
        dfs.append(df)
    dfs = pd.concat(dfs)

    # attribute_num = [0]*(attribute_len+1)
    str_att_num, str_att_rate = 'attribute numbers:', 'attribute rates  :'
    for i in range(attribute_len):
        att_num = dfs['att_%i'%i].sum()
        att_rate = dfs['att_%i'%i].mean()
        str_att_num += '%10d'%(att_num)
        str_att_rate += '%10.6f'%(att_rate)
    print(str_att_num)
    print(str_att_rate)


if __name__ == '__main__':
    input_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels'
    output_txt = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\labels.txt'
    input_txt = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\val.txt'
    # mdet_sta(input_dir, output_txt)
    mdet_sta_val(input_dir, output_txt, input_txt)