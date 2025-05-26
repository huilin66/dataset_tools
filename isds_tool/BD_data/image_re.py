import os

import pandas as pd
from tqdm import tqdm
from skimage import io
from pathlib import Path
from PIL import Image
import numpy as np


def img2png(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_list = os.listdir(input_dir)[580:]
    for img_name in tqdm(img_list):
        png_name = Path(img_name).stem + '.png'
        img_path = os.path.join(input_dir, img_name)
        png_path = os.path.join(output_dir, png_name)
        # img = io.imread(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        if len(img.shape) != 3:
            print(f'{img_name} error with {img[0].shape} and {img[1].shape}')
            img = img[0]
        io.imsave(png_path, img)



if __name__ == '__main__':
    pass
    image_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images'
    image_re_dir = r'E:\data\202502_signboard\data_annotation\annotation_result_merge\images_re'
    img2png(image_dir, image_re_dir)
