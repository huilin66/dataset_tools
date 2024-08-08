import os
import shutil
import pandas as pd
from tqdm import tqdm

def select_from_file(input_dir, ouput_dir, selected_file):
    os.makedirs(ouput_dir, exist_ok=True)
    df = pd.read_csv(selected_file, header=None, index_col=None, names=['path'])
    for idx,row in tqdm(df.iterrows()):
        img_path = row['path']
        img_name = os.path.basename(img_path)
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(ouput_dir, img_name)
        shutil.copyfile(input_path, output_path)


if __name__ == '__main__':
    select_from_file(
        input_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\images',
        ouput_dir=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\images_val',
        selected_file=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\val.txt'
    )