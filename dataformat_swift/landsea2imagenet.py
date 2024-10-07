from skimage import io
import os
import shutil
import pandas as pd
from tqdm import tqdm

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

if __name__ == '__main__':
    pass
    root_dir = r'E:\data\tp\data_landsea'
    img_dir = os.path.join(root_dir, 'img')
    output_dir = os.path.join(root_dir, 'dataset')
    new_dir = os.path.join(root_dir, 'dataset_new')
    scene_path = os.path.join(root_dir, 'scene.csv')
    split_path = os.path.join(root_dir, 'split_data_landsea.csv')
    # landsea2imagenet(scene_path, split_path, img_dir, output_dir)
    tif2png(output_dir, new_dir)