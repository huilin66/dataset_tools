import os

import numpy as np
from PIL import Image


def are_images_identical_numpy(image1_path, image2_path):
    """
    使用 NumPy 比较两张图片是否完全相同
    """
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))

    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        return False

    # 直接比较像素值
    return np.array_equal(img1, img2)


def img_compares(img_dir1, img_dir2):
    img_list = os.listdir(img_dir1)
    for img_name in img_list:
        img_path1 = os.path.join(img_dir1, img_name)
        img_path2 = os.path.join(img_dir2, img_name)
        result = are_images_identical_numpy(img_path1, img_path2)
        print(img_name, ':', result)

if __name__ == '__main__':
    pass
    img_dir1 = r'E:\data\202502_signboard\PS\20250616\rectified_image\selected_img_filter1'
    img_dir2 = r'E:\data\202502_signboard\PS\20250616\rectified_image2\selected_img_filter1'
    img_compares(img_dir1, img_dir2)