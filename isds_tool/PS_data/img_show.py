import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_first_matching_images(root_folder, target_filename='1.jpg'):
    """
    读取6个子文件夹中第一个同名图像并按2×3排列显示

    参数：
        root_folder: 包含6个子文件夹的根目录
        target_filename: 要查找的同名文件名（默认'1.jpg'）
    """
    # 获取所有子文件夹路径
    subfolders = [os.path.join(root_folder, f) for f in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, f))]

    # 只取前6个子文件夹（按字母顺序）
    subfolders = sorted(subfolders)[:6]
    if len(subfolders) < 6:
        raise ValueError(f"需要至少6个子文件夹，当前找到 {len(subfolders)} 个")

    # 收集所有找到的图像路径
    image_paths = []
    for folder in subfolders:
        # 查找目标文件
        target_path = os.path.join(folder, target_filename)
        if not os.path.exists(target_path):
            available_files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            if available_files:
                target_path = os.path.join(folder, sorted(available_files)[0])
            else:
                raise FileNotFoundError(f"{folder} 中没有找到图像文件")

        image_paths.append(target_path)

    # 读取并调整所有图像
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR转RGB
        images.append(img)

    # 创建2×3子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Displaying first '{target_filename}' from 6 subfolders", fontsize=16)

    # 显示每个图像
    for idx, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img)
        ax.set_title(f"Folder {idx + 1}\n{os.path.basename(os.path.dirname(image_paths[idx]))}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    root_folder = r"Y:\ZHL\isds\PS\task0625\Kowloon_stationary\rectified_images"  # 替换为你的根目录路径
    display_first_matching_images(root_folder)