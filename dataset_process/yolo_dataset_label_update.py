import os
import shutil

import pandas as pd
from tqdm import tqdm
from natsort import natsorted

def label_update_by_cliped_img(input_dir, output_dir, img_crop_dir, class_path=None, class_list=None):
    if class_path is not None:
        df = pd.read_csv(class_path, header=None, index_col=None, names=['category'])
        cats = df['category'].to_dict()
    elif class_list is not None:
        cats = {i: name for i, name in enumerate(class_list)}
    else:
        ValueError('class_path or class_list must be specified')

    cats2id = {name: id for id, name in cats.items()}
    os.makedirs(output_dir, exist_ok=True)
    for label_name in tqdm(os.listdir(input_dir), desc='Label copy'):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        shutil.copy(input_path, output_path)

    for cat_id, cat_name in cats.items():
        cat_csv_path = os.path.join(img_crop_dir, f'{cat_name}.csv')
        df_cat = pd.read_csv(cat_csv_path, header=0, index_col=0)
        df_cat_differ = df_cat[df_cat['updated_cat_name'] != df_cat['cat_name']]
        df_cat_differ['updated_cat_id'] = df_cat_differ['updated_cat_name'].map(cats2id)
        label_differ_list = df_cat_differ['file_name'].unique()
        for label_differ in tqdm(label_differ_list, desc='Label update'):
            df_label_differ = df_cat_differ[df_cat_differ['file_name'] == label_differ]
            label_path = os.path.join(output_dir, f'{label_differ}.csv')
            df_label = pd.read_csv(label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'w', 'h'], sep=' ')
            for idx, row in df_label_differ.iterrows():
                object_id = row['object_id']
                updated_cat_id = row['updated_cat_id']
                df_label.loc[object_id, 'cat_id'] = cats2id[updated_cat_id]
            df_label.to_csv(label_path, header=False, index=False, sep=' ')
        print(f'updated {len(label_differ_list)} files, {len(df_cat_differ)} boxes')


def get_csv_by_cliped_img(label_dir, img_crop_dir, class_path=None, class_list=None):
    if class_path is not None:
        df = pd.read_csv(class_path, header=None, index_col=None, names=['category'])
        cats = df['category'].to_dict()
    elif class_list is not None:
        cats = {i: name for i, name in enumerate(class_list)}
    else:
        ValueError('class_path or class_list must be specified')

    label_list = os.listdir(label_dir)
    label_list = natsorted(label_list)
    df_label_list = []
    for label_name in tqdm(label_list, desc='Loading labels'):
        label_path = os.path.join(label_dir, label_name)
        df_label = pd.read_csv(label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'w', 'h'], sep=' ')
        df_label['file_name'] = label_name
        df_label['object_id'] = df_label.index
        df_label_list.append(df_label)
    if len(df_label_list) > 0:
        df = pd.concat(df_label_list, axis=0)
        df['cat_name'] = df['cat_id'].map(cats)
        df['updated_cat_name'] = df['cat_name']

        for cat_id in range(len(cats)):
            df_cat = df[df['cat_id'] == cat_id]
            df_cat_path = os.path.join(img_crop_dir, cats[cat_id]+'.csv')
            df_cat.to_csv(df_cat_path, index=False)



if __name__ == '__main__':
    pass
    root_dir = r'E:\data\2024_defect\2024_defect_pure_yolo_final\wall-defect-ogum1-3wsxo\train'
    label_dir = os.path.join(root_dir, 'labels_det')
    label_updated_dir = os.path.join(root_dir, 'labels_det_updated')
    img_crop_dir = os.path.join(root_dir, 'img_crop')
    class_path = os.path.join(root_dir, 'class.txt')
    get_csv_by_cliped_img(label_dir, img_crop_dir, class_path)

    # label_update_by_cliped_img(label_dir, label_updated_dir, img_crop_dir, class_path)