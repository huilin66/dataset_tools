import os

import pandas as pd
from tqdm import tqdm
from skimage import io
from pathlib import Path

def img2png(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_list = os.listdir(input_dir)
    for img_name in tqdm(img_list):
        png_name = Path(img_name).stem + '.png'
        img_path = os.path.join(input_dir, img_name)
        png_path = os.path.join(output_dir, png_name)
        img = io.imread(img_path)
        if len(img.shape) != 3:
            print(f'{img_name} error with {img[0].shape} and {img[1].shape}')
            img = img[0]
        io.imsave(png_path, img)


def img_del(input_dir, csv_path1, csv_path2):
    df1 = pd.read_csv(csv_path1, header=None, index_col=None)
    df2 = pd.read_csv(csv_path2, header=None, index_col=None)
    df = pd.concat([df1, df2], axis=0)
    file_list = df[0].to_list()
    print(len(file_list))
    all_list = os.listdir(input_dir)
    count = 0
    for file_name in all_list:
        if file_name not in file_list:
            count += 1
            os.remove(os.path.join(input_dir, file_name))
    print(count)


if __name__ == '__main__':
    pass
    input_dir = r'E:\data\202502_signboard\annotation_result_merge\images_re'
    output_dir = r'E:\data\202502_signboard\annotation_result_merge\semantic_masks'
    csv_path1 = r'E:\data\202502_signboard\annotation_result_merge\train.txt'
    csv_path2 = r'E:\data\202502_signboard\annotation_result_merge\val.txt'
    # img2png(input_dir, output_dir)
    img_del(output_dir, csv_path1, csv_path2)