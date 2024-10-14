from skimage import io
import os
import shutil
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from collections import namedtuple
Cls = namedtuple('cls1', ['name', 'id', 'color'])
Clss_seg_f2_EN = [
    Cls('background', 0, (0, 0, 0)),
    Cls('orchard', 1, (0, 255, 127)),
    Cls('farmland', 2, (190, 250, 200)),
    Cls('paddy', 3, (127, 255, 212)),
    Cls('aquafarm', 4, (255, 174, 136)),
    Cls('greenhouse', 5, (220, 174, 136)),
    Cls('fish raft', 6, (255, 100, 136)),
    Cls('marine cage', 7, (255, 174, 100)),
    Cls('float ball', 8, (220, 100, 100)),
    Cls('sand beach', 9, (128, 64, 64)),
    Cls('mud beach', 10, (100, 50, 50)),
    Cls('rock beach', 11, (150, 75, 75)),
    Cls('low rural residence', 12, (128, 0, 255)),
    Cls('low town residence', 13, (200, 50, 205)),
    Cls('low urban residence', 14, (128, 100, 155)),
    Cls('high urban residence', 15, (200, 50, 205)),
    Cls('residential podium', 16, (150, 50, 255)),
    Cls('community services building', 17, (255, 0, 0)),
    Cls('sports field', 18, (225, 0, 0)),
    Cls('teaching building', 19, (200, 0, 0)),
    Cls('municipal building', 20, (175, 0, 0)),
    Cls('low commercial building', 21, (100, 100, 100)),
    Cls('high commercial building', 22, (120, 120, 120)),
    Cls('commercial building podium', 23, (200, 200, 200)),
    Cls('low industrial building', 24, (255, 200, 255)),
    Cls('high industrial building', 25, (220, 200, 220)),
    Cls('field road', 26, (250, 250, 250)),
    Cls('lane', 27, (230, 230, 230)),
    Cls('roadway', 28, (210, 210, 210)),
    Cls('highway', 29, (180, 180, 180)),
    Cls('parking lot', 30, (150, 150, 150)),
    Cls('square', 31, (130, 130, 130)),
    Cls('natural forest', 32, (0, 100, 0)),
    Cls('greening forest', 33, (0, 150, 0)),
    Cls('natural grassland', 34, (128, 200, 64)),
    Cls('greening grassland', 35, (0, 255, 0)),
    Cls('sea water', 36, (119, 187, 255)),
    Cls('land water', 37, (100, 150, 255)),
    Cls('ship', 38, (0, 64, 128)),
    Cls('breakwater', 39, (198, 198, 140)),
    Cls('dam', 40, (175, 198, 140)),
    Cls('port', 41, (198, 175, 140)),
    Cls('bareland', 42, (255, 255, 0)),
    Cls('vacant lot', 43, (255, 255, 200)),
    Cls('landslide', 44, (255, 225, 0)),
    Cls('reef', 45, (225, 255, 0)),
    Cls('island', 46, (255, 225, 200)),
    Cls('peak', 47, (225, 255, 200)),
]

def get_info(scene_path, split_path):
    df_scene = pd.read_csv(scene_path, header=0, index_col=0)
    df_split = pd.read_csv(split_path, header=0, index_col=0)
    df_scene = df_scene[['file_name', 'scene']]
    df_split = df_split[['file_name', 'split']]
    df_info = pd.merge(df_scene, df_split, on='file_name', how='inner')
    return df_info


def landsea2imagenet(scene_path, split_path, img_dir, output_dir):
    df_info = get_info(scene_path, split_path)
    scenes_list = df_info['scene'].unique()
    for scene in scenes_list:
        train_scene_dir = os.path.join(output_dir, 'train', '%s'%scene)
        val_scene_dir = os.path.join(output_dir, 'val', '%s'%scene)
        os.makedirs(train_scene_dir, exist_ok=True)
        os.makedirs(val_scene_dir, exist_ok=True)
    print(scenes_list)
    for idx, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):
        img_path = os.path.join(img_dir, row['file_name'])
        if row['split'] == 'train':
            output_path = os.path.join(output_dir, 'train', '%s'%row['scene'], row['file_name'])
        elif row['split'] == 'val':
            output_path = os.path.join(output_dir, 'val', '%s' % row['scene'], row['file_name'])
        else:
            print(row['file_name'])
        shutil.copy(img_path, output_path)

def tif2png(input_dir, output_dir):
    input_train_dir = os.path.join(input_dir, 'train')
    input_val_dir = os.path.join(input_dir, 'val')
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    for scene in os.listdir(input_train_dir):
        input_train_scene_dir = os.path.join(input_train_dir, scene)
        output_train_scene_dir = os.path.join(output_train_dir, scene)
        os.makedirs(output_train_scene_dir, exist_ok=True)
        for file_name in os.listdir(input_train_scene_dir):
            input_file = os.path.join(input_train_scene_dir, file_name)
            output_file = os.path.join(output_train_scene_dir, file_name.replace('.tif', '.png'))
            img = io.imread(input_file)
            io.imsave(output_file, img)
        input_val_scene_dir = os.path.join(input_val_dir, scene)
        output_val_scene_dir = os.path.join(output_val_dir, scene)
        os.makedirs(output_val_scene_dir, exist_ok=True)
        for file_name in os.listdir(input_val_scene_dir):
            input_file = os.path.join(input_val_scene_dir, file_name)
            output_file = os.path.join(output_val_scene_dir, file_name.replace('.tif', '.png'))
            img = io.imread(input_file)
            io.imsave(output_file, img)

def landsea2seg(split_path, img_dir, gt_dir, output_dir):
    df_info = get_info(scene_path, split_path)

    train_img_dir = os.path.join(output_dir, 'train', 'img')
    train_gt_dir = os.path.join(output_dir, 'train', 'gt')
    val_img_dir = os.path.join(output_dir, 'val', 'img')
    val_gt_dir = os.path.join(output_dir, 'val', 'gt')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_gt_dir, exist_ok=True)

    for idx, row in tqdm(df_info.iterrows(), total=df_info.shape[0]):
        img_path = os.path.join(img_dir, row['file_name'])
        gt_path = os.path.join(gt_dir, row['file_name'])
        if row['split'] == 'train':
            output_img_path = os.path.join(train_img_dir, row['file_name'].replace('.tif', '.png'))
            output_gt_path = os.path.join(train_gt_dir, row['file_name'].replace('.tif', '.png'))
        elif row['split'] == 'val':
            output_img_path = os.path.join(val_img_dir, row['file_name'].replace('.tif', '.png'))
            output_gt_path = os.path.join(val_gt_dir, row['file_name'].replace('.tif', '.png'))
        else:
            print(row['file_name'])
        img = io.imread(img_path)
        io.imsave(output_img_path, img)
        gt_img = io.imread(gt_path)
        io.imsave(output_gt_path, gt_img)

if __name__ == '__main__':
    pass
    root_dir = r'E:\data\tp\data_landsea'
    img_dir = os.path.join(root_dir, 'img')
    gt_dir = os.path.join(root_dir, 'gt')
    output_dir = os.path.join(root_dir, 'dataset')
    new_dir = os.path.join(root_dir, 'dataset_new')
    scene_path = os.path.join(root_dir, 'scene.csv')
    split_path = os.path.join(root_dir, 'split_data_landsea.csv')
    # landsea2imagenet(scene_path, split_path, img_dir, output_dir)
    # tif2png(output_dir, new_dir)

    # landsea2seg(split_path, img_dir, gt_dir, output_dir)

    name_list = [cls.name for cls in Clss_seg_f2_EN]
    color_list = [cls.color for cls in Clss_seg_f2_EN]
    print(name_list)
    print(color_list)