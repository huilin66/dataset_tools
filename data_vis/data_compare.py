import os
import cv2
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from skimage import io
import matplotlib.pyplot as plt


keep_nums1 = [
    740, 756, 764, 768, 780, 828, 832, 856, 860, 862, 1044, 1050, 1072, 1112, 1152, 1312, 1330, 1342, 1346, 1364, 1372,
    1382, 1430, 1672, 1700, 1736, 1930, 1950, 2036, 2038,
]
keep_nums2 = [
    808, 900, 930, 1076, 1146, 1244, 1258, 1304, 1356, 1506, 1588, 1622, 1642, 1756, 1778, 1798, 1800, 1832, 1970,
]

def metric_compare():
    # 示例医疗数据：不同指标的两个数据系列的数值
    data = pd.DataFrame({
        'Indicator': ['P', 'R', 'mAP50', 'mAP50-95'],
        'yolo9': [0.957, 0.812, 0.875, 0.701],
        'yolo8': [0.937, 0.797, 0.88, 0.741],
        'tfd': [0.938, 0.82, 0.88, 0.706],
        'ema2': [0.952, 0.84, 0.887, 0.71],
        'ema3': [0.936, 0.852, 0.888, 0.711],
    })

    # 创建雷达图
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")

    # 设置雷达图的角度和标签
    categories = list(data['Indicator'])
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 绘制雷达图的轴线
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_ylim(0.7, 1)
    # ax.set_yticks(np.arange(0, 1, 10))


    # 使用亮色绘制两个数据系列
    values_series1 = list(data['tfd'])
    values_series1 += values_series1[:1]
    ax.fill(angles, values_series1, 'm', alpha=0.3, label='Model 1')

    values_series2 = list(data['ema2'])
    values_series2 += values_series2[:1]
    ax.fill(angles, values_series2, 'c', alpha=0.3, label='Model 2')

    values_series3 = list(data['ema3'])
    values_series3 += values_series3[:1]
    ax.fill(angles, values_series3, 'r', alpha=0.3, label='Model 3')


    values_series4 = list(data['yolo9'])
    values_series4 += values_series4[:1]
    ax.fill(angles, values_series4, 'g', alpha=0.3, label='Model 4')

    values_series5 = list(data['yolo8'])
    values_series5 += values_series5[:1]
    ax.fill(angles, values_series5, 'b', alpha=0.3, label='Model 5')
    # 添加标题和图例
    plt.title("metrics comparison", fontsize=16)
    plt.legend(loc='upper right', title="Data Series")

    # 显示雷达图
    plt.show()

def cp_val_data(input_dir, output_dir, txt_path):
    os.makedirs(output_dir, exist_ok=True)
    file_list = pd.read_csv(txt_path, header=None, index_col=None)[0].tolist()
    for file_path in tqdm(file_list):
        input_path = os.path.join(input_dir, os.path.basename(file_path))
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        shutil.copyfile(input_path, output_path)

def cat_img(pre_path, gt_path):
    pre_img = cv2.imread(pre_path)
    gt_img = cv2.imread(gt_path)
    cat_img = np.concatenate((pre_img, gt_img), axis=1)
    return cat_img

def cat_show(pre_dir, gt_dir, cat_dir):
    def pre2gt_name(pre_name):
        gt_name = pre_name
        return gt_name
    file_list = os.listdir(pre_dir)
    for file_name in tqdm(file_list):
        pre_path = os.path.join(pre_dir, file_name)
        gt_path = os.path.join(gt_dir, pre2gt_name(file_name))
        cat_path = os.path.join(cat_dir, file_name)
        img_cat = cat_img(pre_path, gt_path)
        cv2.imwrite(cat_path, img_cat)

def cat_imgs(pre_path_list):
    pre_img_list = [io.imread(pre_path) for pre_path in pre_path_list]
    cat_img = np.concatenate(pre_img_list, axis=1)
    return cat_img

def cat_compare(pre_dir_list, show_dir):
    os.makedirs(show_dir, exist_ok=True)
    img_list = os.listdir(pre_dir_list[0])
    for img_name in tqdm(img_list):
        pre_path_list = [os.path.join(pre_dir, img_name) for pre_dir in pre_dir_list]
        cat_img = cat_imgs(pre_path_list)
        io.imsave(os.path.join(show_dir, img_name), cat_img)

if __name__ == '__main__':
    pass
    cat_compare(
        pre_dir_list = [
            r'E:\cp_dir\temp\predictss\yolo8',
            r'E:\cp_dir\temp\predictss\yolo9',
            r'E:\cp_dir\temp\predictss\yolo10',
            r'E:\cp_dir\temp\predictss\mayolo',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\images_vis',
        ],
        show_dir=r'E:\cp_dir\temp\predictss\cat_show',
    )