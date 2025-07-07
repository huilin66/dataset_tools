import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_img_list(img_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list_100 = img_list[:100]
    img_list_100 = [os.path.join(img_dir, img_name) for img_name in img_list_100]
    return img_list_100

def find_closest_row(df, column_name, target):
    idx = (df[column_name] - target).abs().idxmin()
    closest_row = df.loc[idx]
    return closest_row


def get_gps_list(gps_path, img_lists, dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(gps_path, header=0, index_col=None, sep=' ')
    gps_list = []
    for img_path in tqdm(img_lists):
        img_timestamp = int(Path(img_path).stem.split('_')[-1])
        cloest_row = find_closest_row(df, '#timestamp', img_timestamp)
        timestamp = cloest_row['#timestamp']
        dst_img_path = os.path.join(dst_dir, f'{timestamp:.0f}.jpg')
        if not os.path.exists(dst_img_path):
            shutil.copy(img_path, dst_img_path)
            gps_list.append(cloest_row)
    df = pd.concat(gps_list, axis=1).T

    df['#timestamp'] = df['#timestamp'].apply(lambda x: f"{x:.0f}")
    df['latitude'] = df['latitude'].apply(lambda x: f"{x:.7f}")
    df['longitude'] = df['longitude'].apply(lambda x: f"{x:.7f}")
    df['altitude'] = df['altitude'].apply(lambda x: f"{x:.3f}")

    # formatters = {
    #     '#timestamp': lambda x: str(f"{x:.0f}"),
    #     'latitude': lambda x: f"{x:.7}",
    #     'longitude': lambda x: f"{x:.8}",
    #     'altitude': lambda x: f"{x:.3f}",
    # }
    df.to_csv(dst_dir+'_gps.csv', index=False, header=True)

if __name__ == '__main__':
    img_dir = r'Y:\ZHL\isds\PS\task0612\Route1_Kowloon_2025_06_10-13_01_37\Route1_Kowloon_2025_06_10-13_01_37\rectified_image\cam_DA4930148_select'
    dst_dir = r'E:\data\202502_signboard\PS\20250612\demo1\img'
    gps_path = r'Y:\ZHL\isds\PS\task0612\Route1_Kowloon_2025_06_10-13_01_37\Route1_Kowloon_2025_06_10-13_01_37\gps_data_20250610130137799_20250610131300074.txt'

    img_list = get_img_list(img_dir)
    get_gps_list(gps_path, img_list, dst_dir)






