import os
from pathlib import Path

import PIL
import cv2
import numpy as np
from skimage import io
from PIL import Image

def yolo_to_semantic_segmentation(txt_path, image_path, output_dir, class_mapping=None, with_att=False):
    """
    将YOLO实例分割标签转换为语义分割掩码
    参数：
        txt_path: YOLO标签文件路径（包含多边形坐标）
        image_path: 对应图像路径（用于获取尺寸）
        output_dir: 输出掩码目录
        class_mapping: 类别ID映射字典（可选，用于重映射类别）
    返回：
        无，直接保存掩码文件
    """
    # 读取图像获取尺寸
    image = io.imread(image_path)
    if len(image.shape) != 3:
        print(f'{image_path} shape error {image.shape}')
        image = image[0]
    height, width = image.shape[:2]

    # 初始化全零掩码（单通道）
    semantic_mask = np.zeros((height, width), dtype=np.uint8)

    # 读取并解析YOLO标签
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        # 解析类别和归一化坐标（YOLO格式：[class x_center y_center w h]）
        class_id = int(parts[0])
        if class_mapping and class_id in class_mapping:
            class_id = class_mapping[class_id]  # 重映射类别ID
        if with_att:
            att_len = int(parts[1])
            atts = parts[2:att_len+2]
            idx_seg = att_len+2
        else:
            idx_seg = 1
        points = list(map(float, parts[idx_seg:]))

        # 将坐标转换为整数像素位置
        polygon = []
        for i in range(0, len(points), 2):
            x = int(points[i] * width)
            y = int(points[i + 1] * height)
            polygon.append([x, y])

        # 绘制多边形到掩码
        if len(polygon) >= 3:  # 确保至少有3个点
            cv2.fillPoly(semantic_mask, [np.array(polygon, dtype=np.int32)], color=class_id)

    # 生成输出路径（与图像同目录，扩展名.png）
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{image_basename}.png")

    # 保存掩码（PNG格式保留透明度，如需JPG需二值化）
    # cv2.imwrite(mask_path, semantic_mask)
    pil_image = Image.fromarray(semantic_mask)
    pil_image.save(mask_path)
    print(f"Saved semantic mask to {mask_path}")

def process_dataset(yolo_root, images_dir="images", output_dir="semantic_masks", with_att=False):
    """
    批量处理整个数据集
    参数：
        yolo_root: YOLO数据集根目录（包含images和labels子目录）
        images_dir: 图像目录名称（默认为'images'）
        output_dir: 输出掩码目录
    """
    labels_dir = os.path.join(yolo_root, "labels")
    images_dir = os.path.join(yolo_root, images_dir)

    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        txt_name = Path(img_name).stem + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        image_path = os.path.join(images_dir, img_name)

        if os.path.exists(image_path):
            yolo_to_semantic_segmentation(txt_path, image_path, output_dir, with_att=with_att)
        else:
            print(f"警告：图像缺失 {image_path}")





def find_mismatched_files(folder_a, folder_b):

    files_in_a = os.listdir(folder_a)
    files_in_b1 = os.listdir(folder_b)
    files_in_b = [file_name.replace('_seg.png', '.png') for file_name in os.listdir(folder_b)]

    files_in_a = set(files_in_a)
    files_in_b = set(files_in_b)

    only_in_a = files_in_a - files_in_b

    # 找出只存在于文件夹b中的文件
    only_in_b = files_in_b - files_in_a

    # 输出不符合条件的路径
    print("只存在于文件夹a中的文件:")
    for file in only_in_a:
        print(os.path.join(folder_a, file))

    print("\n只存在于文件夹b中的文件:")
    for file in only_in_b:
        print(os.path.join(folder_b, file))



if __name__ == "__main__":
    yolo_root = r'E:\data\202502_signboard\annotation_result_merge'
    output_dir = r'E:\data\202502_signboard\annotation_result_merge\semantic_masks'
    process_dataset(yolo_root, images_dir="images_re", output_dir=output_dir, with_att=True)


    # find_mismatched_files(
    #     r'E:\data\202502_signboard\annotation_result_merge\images_re',
    #     r'E:\data\202502_signboard\annotation_result_merge\semantic_masks'
    # )

    # yolo_root = r"E:\data\202502_signboard\annotation_result_merge"
    # img_path = r'E:\data\202502_signboard\annotation_result_merge\images\DSC04970.JPG'
    # print(os.path.exists(img_path))
    # img1 = cv2.imread(img_path)
    # print(img1)
    # img2 = PIL.Image.open(img_path)
    # print(img2)
    # img3 = io.imread(img_path)
    # print(img3)
    # print()