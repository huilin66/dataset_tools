import os
import json
import shutil

from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

def random_split_class(input_dir, output_dir, train_ratio=0.9, random_seed=1010):
    pass
    level_list= os.listdir(input_dir)
    if 'train' in level_list:
        level_list.remove('train')
    if 'val' in level_list:
        level_list.remove('val')
    for level_name in level_list:
        level_dir = os.path.join(input_dir, level_name)
        level_dir_train = os.path.join(output_dir, 'train', level_name)
        level_dir_val = os.path.join(output_dir, 'val', level_name)
        os.makedirs(level_dir_train, exist_ok=True)
        os.makedirs(level_dir_val, exist_ok=True)

        file_list = os.listdir(level_dir)
        np.random.seed(random_seed)
        np.random.shuffle(file_list)
        train_num = int(len(file_list) * train_ratio)
        for idx, file_name in enumerate(tqdm(file_list)):
            src_path = os.path.join(level_dir, file_name)
            if idx<train_num:
                dst_path = os.path.join(level_dir_train, file_name)
            else:
                dst_path = os.path.join(level_dir_val, file_name)
            shutil.copyfile(src_path, dst_path)
        print(f'\ncopy {train_num} files to {level_dir_train}'
              f'\ncopy {len(file_list) - train_num} files to {level_dir_val}\n')


def ref_split_class(ref_path, info_path, input_dir, output_dir=None,
                    defect_list=['abandonment', 'broken', 'corrosion', 'deformation'],
                    level_list = ['no', 'medium', 'high']):

    if output_dir is None:
        output_dir = input_dir
    for defect in defect_list:
        train_dir = os.path.join(output_dir, defect, 'train')
        val_dir = os.path.join(output_dir, defect, 'val')
        for level in level_list:
            defect_train_dir = os.path.join(train_dir, level)
            defect_val_dir = os.path.join(val_dir, level)
            os.makedirs(defect_train_dir, exist_ok=True)
            os.makedirs(defect_val_dir, exist_ok=True)


    df_info = pd.read_csv(info_path, header=0, index_col=0)
    df_info['split'] = 'train'

    df_ref = pd.read_csv(ref_path, header=None, index_col=None, names=['image_name'])
    df_ref['file_name'] = df_ref['image_name'].apply(lambda x: Path(x).stem)

    df_info.loc[df_info['image_x'].isin(df_ref['file_name']), 'split'] = 'val'

    for idx, row in tqdm(df_info.iterrows(),total=len(df_info)):
        split = row['split']
        object_name = row['object_name_full']
        for defect in defect_list:
            defect_value = int(row[defect])
            defect_level = level_list[defect_value]
            src_path = os.path.join(output_dir, defect, defect_level, object_name)
            dst_path = os.path.join(output_dir, defect, split, defect_level, object_name)

            shutil.copy(src_path, dst_path)


def all2category(input_dir, output_dir, csv_path, random_seed=1010):
    pass


if __name__ == '__main__':
    pass
    # risk_a_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/abandonment'
    # risk_b_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/broken'
    # risk_c_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/corrosion'
    # risk_d_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/deformation'
    # random_split_class(risk_a_dir, risk_a_dir)
    # random_split_class(risk_b_dir, risk_b_dir)
    # random_split_class(risk_c_dir, risk_c_dir)
    # random_split_class(risk_d_dir, risk_d_dir)
    defect_list = ['deformation', 'broken', 'abandonment', 'corrosion']
    level_list = ['no', 'medium', 'high']
    dataset_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618'
    sta_dir = os.path.join(dataset_dir, "labels_sta")
    info_path = os.path.join(sta_dir, "info.csv")
    image_crop_dir = os.path.join(dataset_dir, 'images_crop')
    val_path = os.path.join(dataset_dir, 'val.txt')
    ref_split_class(ref_path=val_path,
                    info_path=info_path,
                    input_dir=image_crop_dir,
                    defect_list=defect_list,
                    level_list=level_list)

    # ref_split_class(ref_path=r'/localnvme/data/billboard/demo_data/data744_mseg_c6/val.txt',
    #                 info_path=r'/localnvme/data/billboard/demo_data/data744_mseg_c6/labels_sta/info.csv',
    #                 input_dir=r'/localnvme/data/billboard/demo_data/data744_mseg_c6/images_crop/abandonment',
    #                 images_crop_revert_path=r'/localnvme/data/billboard/demo_data/data744_mseg_c6/images_crop_revert.json')