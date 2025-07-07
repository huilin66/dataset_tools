import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import io
from tqdm import tqdm

def multi_show(model_preds_path, save_path):
    model_preds = [np.load(model_pred_path) for model_pred_path in model_preds_path]


    cmap = mcolors.ListedColormap(['black', 'blue', 'green', 'red'])  # 标签 0 -> 黑色, 1 -> 蓝色, 2 -> 绿色, 3 -> 红色
    bounds = [0, 1, 2, 3, 4]  # 标签边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)


    fig, axs = plt.subplots(1, len(model_preds), figsize=(len(model_preds)*5, 5))  # 创建1行3列的子图

    for i in range(len(model_preds)):
        axs[i].imshow(model_preds[i], cmap=cmap, norm=norm)
        axs[i].axis('off')  # 隐藏坐标轴


    # 3. 调整布局并保存拼接图像
    plt.tight_layout()
    plt.savefig(save_path)  # 保存为图片
    # plt.show()  # 显示拼接后的图像
    plt.close()

def multi_show_dir(model_preds_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_list = os.listdir(model_preds_dir[0])
    for file_name in tqdm(file_list):
        files_path = [os.path.join(model_pred_dir, file_name) for model_pred_dir in model_preds_dir]
        save_path = os.path.join(save_dir, file_name.replace('.npy', '.png'))
        multi_show(files_path, save_path)


def single_show(model_pred_path, save_path=None):
    # segmentation_result = io.imread(model_pred_path)
    segmentation_result = np.load(model_pred_path)

    # 1. 定义颜色映射 (colormap)，映射 0-3 的标签到不同颜色
    cmap = mcolors.ListedColormap(['black', 'blue', 'green', 'red'])  # 0 -> black, 1 -> blue, 2 -> green, 3 -> red
    bounds = [0, 1, 2, 3, 4]  # 标签边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 2. 可视化结果
    plt.imshow(segmentation_result, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2, 3], label='Class labels')
    plt.title('Segmentation Visualization (0-3 Classes)')
    plt.show()

if __name__ == '__main__':
    preds_dir = [
        r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_gt',
        r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_unet_infer_npy',
        r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_ocrnet_infer_npy',
        r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_ocrnet2_infer_npy',
    ]
    save_dir = r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_pred_compare'
    multi_show_dir(preds_dir, save_dir)

    # single_show(r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_ocrnet_infer\000001.jpg')
    # single_show(r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_unet_infer_npy\000001.npy')