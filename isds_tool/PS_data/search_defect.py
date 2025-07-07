import os

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from isds_tool.PS_data.yolo_tools import get_yolo_label_df


def get_trian_list(train_path):
    if train_path is None:
        return None, None
    else:
        df_train = pd.read_csv(train_path, names=['file_name'])
        df_train['file_steam'] = df_train['file_name'].apply(lambda x: Path(x).stem)
        file_name_list = df_train['file_name'].to_list()
        file_stem_list = df_train['file_steam'].to_list()
        return file_name_list, file_stem_list

def search_defect(input_dir, attribute_path, save_path=None, train_path=None):
    if save_path is None:
        save_path = input_dir + '.csv'
    train_name_list, train_stem_list = get_trian_list(train_path)
    file_list = os.listdir(input_dir)

    dfs = []
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        df = get_yolo_label_df(file_path, mdet=True, attributes=attribute_path, all_info=True)
        if train_stem_list is not None:
            file_stem = Path(file_name).stem
            if file_stem in train_stem_list:
                df['split'] = 'train'
            else:
                df['split'] = 'val'
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all.to_csv(save_path, encoding='utf-8-sig')
    print('save to ', save_path)
if __name__ == '__main__':
    input_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0624/images_infer/labels'
    att_path = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618/attribute.yaml'
    save_path = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0624/images_infer.csv'
    train_path = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0624/train.txt'
    search_defect(input_dir, att_path, save_path, train_path)