import shutil

import cv2
import os

import pandas as pd
from tqdm import tqdm
img_dir_list = [
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\corrosion_data\corrosion_data\train\batch1',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\corrosion_data\corrosion_data\train\batch2',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\corrosion_data\corrosion_data\val\batch1',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\deformation_data\deformation_data\train',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\deformation_data\deformation_data\val',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\signboard_train\train',
    r'E:\data\202502_signboard\20250224 Signboard Data and CDU\signboard_val\val'

]
all_csv_path = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\data_all.csv'
dst_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\selected_img_3000'
batch1_1_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\data\batch1_1500'
batch1_2_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\data\batch2_1500'

def get_img_list(img_dir_list):
    total_num = []
    for img_dir in img_dir_list:
        for img_sub_name in os.listdir(img_dir):
            img_sub_dir = os.path.join(img_dir, img_sub_name)
            img_num = len(os.listdir(img_sub_dir))
            total_num.append(img_num)
            print(f"{img_sub_name} --> {img_num}, {os.listdir(img_sub_dir)[0]}")
    print(f"total number of images: {sum(total_num)} in {len(total_num)} dir")

def get_img_sta(img_dir_list, all_csv_path=all_csv_path):
    pass
    count = 1
    df_list = []
    for img_dir in img_dir_list:
        for img_sub_name in os.listdir(img_dir):
            img_sub_dir = os.path.join(img_dir, img_sub_name)
            if not os.path.isdir(img_sub_dir):
                continue
            df = pd.DataFrame(None, columns=['img_name', 'img_dir', 'img_width', 'img_height'])
            for img_name in tqdm(os.listdir(img_sub_dir), desc=f"115:{count}"):
                if img_name.endswith('.jpg') or img_name.endswith('.png') or img_name.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(img_sub_dir, img_name))
                    df.loc[len(df)] = [img_name, img_sub_dir, img.shape[0], img.shape[1]]
            df.to_csv(img_sub_dir+'.csv', encoding='utf-8')
            count += 1
            df_list.append(df)

    df_all = pd.concat(df_list)
    df_all.to_csv(all_csv_path, encoding='utf-8')

def get_imgs(dst_dir, all_csv_path=all_csv_path):
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(all_csv_path, header=0, index_col=0)
    print(f'before drop_duplicates {len(df)}')
    df = df.drop_duplicates(subset=['img_name'], keep='first')
    print(f'after drop_duplicates {len(df)}')
    sampled_df = df.sample(n=3000, replace=False, random_state=42).sort_index()
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc=f"copy file"):
        img_path = os.path.join(row['img_dir'], row['img_name'])
        dst_img_path = os.path.join(dst_dir, f"{row['img_name']}")
        shutil.copyfile(img_path, dst_img_path)

def copy_imgs(dst_dir, batch1_1_dir, batch1_2_dir):
    os.makedirs(batch1_1_dir, exist_ok=True)
    os.makedirs(batch1_2_dir, exist_ok=True)
    img_list = os.listdir(dst_dir)
    for idx, img_name in tqdm(enumerate(img_list), total=len(img_list), desc=f"move file"):
        input_img_path = os.path.join(dst_dir, img_name)
        if idx % 2 == 0:
            output_img_path = os.path.join(batch1_1_dir, img_name)
        else:
            output_img_path = os.path.join(batch1_2_dir, img_name)
        shutil.move(input_img_path, output_img_path)

if __name__ == '__main__':
    pass
    # get_img_list(img_dir_list)
    # get_img_sta(img_dir_list)
    get_imgs(dst_dir)
    copy_imgs(dst_dir, batch1_1_dir, batch1_2_dir)