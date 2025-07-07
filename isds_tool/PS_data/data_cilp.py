import os
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np

def imgs_clip(input_dir, output_dir):
    img_list = os.listdir(input_dir)
    output_left = os.path.join(output_dir, 'left')
    output_right = os.path.join(output_dir, 'right')
    os.makedirs(output_left, exist_ok=True)
    os.makedirs(output_right, exist_ok=True)
    for img_name in tqdm(img_list):
        input_path = os.path.join(input_dir, img_name)
        output_left_path = os.path.join(output_left, Path(img_name).stem + '.png')
        output_right_path = os.path.join(output_right, Path(img_name).stem + '.png')
        split_image_left_right(input_path, output_left_path, output_right_path)

def split_image_left_right(input_path, output_left_path, output_right_path):
    """
    将一张 JPG 图片裁剪成左右两部分并保存。

    :param input_path: 输入图片路径
    :param output_left_path: 左半部分保存路径
    :param output_right_path: 右半部分保存路径
    """
    # 打开图片
    with Image.open(input_path) as img:
        width, height = img.size

        # 计算裁剪点
        left_half = width // 2

        # 裁剪左半部分
        left_img = img.crop((0, 0, left_half, height))

        # 裁剪右半部分
        right_img = img.crop((left_half, 0, width, height))

        # 保存裁剪后的图像
        left_img.save(output_left_path)
        right_img.save(output_right_path)


def split_image_left_right_cv(input_path, output_left_path, output_right_path):
    """
    使用 OpenCV 快速裁剪 JPG 图片为左右两部分
    """
    # 读取图片（BGR 格式）
    img = cv2.imread(input_path)

    # 获取图片尺寸
    height, width = img.shape[:2]

    # 计算裁剪点
    left_half = width // 2

    # 裁剪左半部分
    left_img = img[:, :left_half]  # [行, 列] 切片

    # 裁剪右半部分
    right_img = img[:, left_half:]  # [行, 列] 切片

    # 保存裁剪后的图像（OpenCV 默认 BGR，但 JPG 通常 RGB，所以需要转换）
    cv2.imwrite(output_left_path, cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 6])  # 如果需要 RGB
    cv2.imwrite(output_right_path, cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 6])  # 如果需要 RGB


def img_merge(input_left_path, input_right_path, output_path):
    img_left = cv2.imread(input_left_path)
    img_right = cv2.imread(input_right_path)
    img_output = np.hstack((img_left, img_right))
    cv2.imwrite(output_path, img_output)

def imgs_merge(input_dir_left, input_dir_right, output_dir):
    img_list = os.listdir(input_dir_left)
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        input_img_left_path = os.path.join(input_dir_left, img_name)
        input_img_right_path = os.path.join(input_dir_right, img_name)
        output_img_path = os.path.join(output_dir, img_name)
        split_image_left_right(input_img_left_path, input_img_right_path, output_img_path)

if __name__ == '__main__':
    pass
    imgs_clip(r'E:\data\202502_signboard\data_annotation\task\task0519\images',
              r'E:\data\202502_signboard\data_annotation\task\task0519\images_split')