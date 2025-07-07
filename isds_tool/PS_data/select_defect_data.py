import os
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from yolo_mask_crop import extract_polygon_from_image, image_read, polygon_swift, image_save

def random_select_by_sta(attribute_path, input_dir1, input_dir2, output_dir1, output_dir2):
    pass
    df = pd.read_csv(attribute_path, header=0, index_col=0)
    print(df.head())
    cond = ((df['abandonment'] > 0) | (df['broken'] > 0.0) | (df['corrosion'] > 0.0) | (df['deformation'] > 0.0))
    df_with_defect = df[cond]
    df_with_abandonment = df_with_defect[df_with_defect['abandonment']>0]
    df_with_broken = df_with_defect[df_with_defect['broken']>0]
    df_with_corrosion = df_with_defect[df_with_defect['corrosion']>0]
    df_with_deformation = df_with_defect[df_with_defect['deformation']>0]
    df_with_abandonment_sample = df_with_abandonment.sample(n=10) if df_with_abandonment.shape[0]>10 else df_with_abandonment
    df_with_broken_sample = df_with_broken.sample(n=10)
    df_with_corrosion_sample = df_with_corrosion.sample(n=10)
    df_with_deformation_sample = df_with_deformation.sample(n=10)
    # print(df_with_abandonment_sample['image'].unique())
    defect_list = ['abandonment', 'broken', 'corrosion', 'deformation']
    df_defect_list = [df_with_abandonment_sample, df_with_broken_sample, df_with_corrosion_sample, df_with_deformation_sample]
    for i in range(4):
        defect_name = defect_list[i]
        df_defect = df_defect_list[i]
        output_defect_dir1 = os.path.join(output_dir1, defect_name)
        output_defect_dir2 = os.path.join(output_dir2, defect_name)
        os.makedirs(output_defect_dir1, exist_ok=True)
        os.makedirs(output_defect_dir2, exist_ok=True)
        for idx, row in df_defect.iterrows():
            image_name1 = f"{row['image']}.jpg"
            image_name2 = f"{row['image']}_{idx}.jpg"
            input_path1 = os.path.join(input_dir1, image_name1)
            input_path2 = os.path.join(input_dir2, image_name2)
            output_path1 = os.path.join(output_defect_dir1, image_name2)
            output_path2 = os.path.join(output_defect_dir2, image_name2)
            shutil.copy(input_path1, output_path1)
            shutil.copy(input_path2, output_path2)


def random_select_by_crop(attribute_path, input_dir1, input_dir2, output_dir1, output_dir2):
    image_list = os.listdir(input_dir1)
    image_stem_list = [Path(image_name).stem for image_name in image_list]
    image_name_dict = dict(zip(image_stem_list, image_list))

    df = pd.read_csv(attribute_path, header=0, index_col=0)
    defect_count = {
        'abandonment' : 0,
        'broken' : 0,
        'corrosion' : 0,
        'deformation' : 0
    }
    img_crop_list = os.listdir(input_dir2)
    img_crop_list.sort()
    for img_crop_name in tqdm(img_crop_list):
        img_idx = int(Path(img_crop_name).stem.split('_')[-1])
        img_name_stem = img_crop_name.replace(f"_{img_idx}.png", "")
        row = df.loc[(df.index==img_idx)&(df['image']==img_name_stem)].squeeze()
        defect_list = ['abandonment', 'broken', 'corrosion', 'deformation']
        for defect in defect_list:
            if row[defect] > 0 and defect_count[defect]<15:
                output_defect_dir1 = os.path.join(output_dir1, defect)
                output_defect_dir2 = os.path.join(output_dir2, defect)
                os.makedirs(output_defect_dir1, exist_ok=True)
                os.makedirs(output_defect_dir2, exist_ok=True)
                input_path1 = os.path.join(input_dir1, image_name_dict[img_name_stem])
                input_path2 = os.path.join(input_dir2, img_crop_name)
                output_path1 = os.path.join(output_defect_dir1, img_crop_name)
                output_path2 = os.path.join(output_defect_dir2, img_crop_name)
                shutil.copy(input_path1, output_path1)
                shutil.copy(input_path2, output_path2)
                defect_count[defect] += 1
                print(defect_count)


def select_by_predict(compare_dir, image_dir, output_dir, defect_list, level_list):
    pass
    for defect in defect_list:
        defect_dir_val = os.path.join(output_dir, defect, 'val')
        for level in level_list:
            level_dir = os.path.join(defect_dir_val, level)
            os.makedirs(level_dir, exist_ok=True)
    img_list = os.listdir(image_dir)
    stem2img_dict = {Path(obj_name).stem: obj_name for obj_name in img_list}

    compare_list = os.listdir(compare_dir)
    for compare_name in tqdm(compare_list):
        compare_path = os.path.join(compare_dir, compare_name)
        df_compare = pd.read_csv(compare_path, header=0, index_col=0)
        image_name = stem2img_dict[Path(compare_name).stem]
        image_path = os.path.join(image_dir, image_name)
        image = image_read(image_path)
        for idx, row in df_compare.iterrows():
            if pd.isna(row['id_pred']):
                continue
            obj_id = row['id_labels']
            xys = row['xy_pred']
            save_name = f'{Path(compare_name).stem}_{obj_id}'+Path(image_name).suffix
            polygon_coords = polygon_swift(xys, image)
            img_crop, _, _ = extract_polygon_from_image(image.copy(), polygon_coords, crop_method='without_background_box_shape',)

            if img_crop.shape[0]*img_crop.shape[1]>0:
                for id_defect, defect_name in enumerate(defect_list):
                    defect_label = int(row[f'att{id_defect}_labels'])
                    defect_level = level_list[defect_label]
                    save_path = os.path.join(output_dir, defect_name, 'val', defect_level, save_name)
                    image_save(save_path, img_crop)


def obj_record(attribute_path, box_path, obj2img_path=None, info_path=None):
    pass

    if info_path is None:
        info_path = os.path.join(os.path.dirname(attribute_path), 'info.csv')
    df_att = pd.read_csv(attribute_path, header=0, index_col=0)
    df_att['object_name'] = df_att.apply(lambda row: f"{row['image']}_{row.name}", axis=1)
    df_att = df_att.sort_values(by=['object_name'])

    df_box = pd.read_csv(box_path, header=0, index_col=0)
    df_box['object_name'] = df_box.apply(lambda row: f"{row['image']}_{row.name}", axis=1)
    df_box = df_box.sort_values(by=['object_name'])

    df_info = pd.merge(left=df_att, right=df_box, how='left', on='object_name')

    if obj2img_path is not None:
        with open(obj2img_path, 'rb') as f:
            obj2img_dict = json.load(f)
        obj_list = list(obj2img_dict.keys())
        stem2obj_dict = {Path(obj_name).stem: obj_name for obj_name in obj_list}
        df_info['object_name_full'] = df_info['object_name'].apply(lambda x: stem2obj_dict.get(x))
    df_info['area'] = df_info['width'] * df_info['height']
    df_info['small_obj'] = df_info['area']<0.01
    df_info.to_csv(info_path, encoding='utf-8-sig')
    print(f'merge {attribute_path}+{box_path} --> {info_path} finished!')

if __name__ == '__main__':
    pass
    # attribute_path = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels_sta/sta_attribute.csv'
    # random_select_by_sta(attribute_path,
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/images',
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/images_crop',
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/demo_defect/ps_data/images',
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/demo_defect/ps_data/images_vis',
    #                      )


    # attribute_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/sta_attribute.csv'
    # random_select_by_crop(attribute_path,
    #          r'/localnvme/data/billboard/bd_data/data626_mseg_c6/images',
    #          r'/localnvme/data/billboard/bd_data/data626_mseg_c6/images_crop',
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/demo_defect/bd_data/images',
    #          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/demo_defect/bd_data/images_vis',
    #                      )


    # attribute_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/sta_attribute.csv'
    # box_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/sta_box.csv'
    # info_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/info.csv'
    # obj_record(attribute_path, box_path, info_path)
    # attribute_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/sta_attribute.csv'
    # box_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/sta_box.csv'
    # info_path = r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_sta/info.csv'
    # obj_record(attribute_path, box_path, info_path)

    dataset_dir = r'/localnvme/data/billboard/fused_data/data1422_mseg_c6'
    gt_dir = os.path.join(dataset_dir, "labels")
    sta_dir = os.path.join(dataset_dir, "labels_sta")
    obj2img_path = os.path.join(dataset_dir, "images_crop.json")
    sta_att_path = os.path.join(sta_dir, "sta_attribute.csv")
    box_path = os.path.join(sta_dir, "sta_box.csv")
    info_path = os.path.join(sta_dir, "info.csv")
    obj_record(sta_att_path, box_path, obj2img_path, info_path)

    # image_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618/images'
    # compare_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618/labels_pred_compare'
    # output_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618/predict_val'
    # select_by_predict(compare_dir, image_dir, output_dir,
    #                   defect_list = ['abandonment', 'broken', 'corrosion', 'deformation'],
    #                   level_list = ['no', 'medium', 'high'])