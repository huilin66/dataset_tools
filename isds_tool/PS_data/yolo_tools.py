import os
import shutil
import sys
from tqdm import tqdm
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from scipy import stats
import pandas as pd
import yaml
from sklearn.model_selection import KFold

sys.path.append(r'E:\repository\dataset_tools')

ATT_FILE = r'/localnvme/data/billboard/attribute.yaml'
ATT_L2_FILE = r'/localnvme/data/billboard/attribute_l2.yaml'
CLASS_C6_FILE = r'/localnvme/data/billboard/class_c6.txt'
CLASS_C5_FILE = r'/localnvme/data/billboard/class_c5.txt'

def get_stem2img(img_dir):
    img_list = os.listdir(img_dir)
    stem2img = {}
    for img_name in img_list:
        stem = Path(img_name).stem
        stem2img[stem] = img_name
    return stem2img


def mseg2seg_gt(input_dir, output_dir):
    label_list = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                attribute_num = int(num_list[1])
                num_list_seg = num_list[:1]+num_list[1+attribute_num+1:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)

def mseg2seg(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list, desc='mseg2seg:'):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                attribute_num = int(num_list[1])
                num_list_seg = num_list[:1]+num_list[1+attribute_num+1:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)


def mseg_class_update(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list, desc='change to c6:'):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def mseg_class_update2(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list, desc='change to c5:'):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '4':
                    lines[idx] = str(int(lines[idx][0])-2)+lines[idx][1:]
                else:
                    continue
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def mseg_class_update_gt(input_label_dir, output_label_dir, cp_img=True):
    label_list = os.listdir(input_label_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)

def mseg_attribute_update_gt(input_label_dir, output_label_dir):
    label_list = os.listdir(input_label_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] != '6':
                    continue
                else:
                    lines[idx] = '6 4 0 0 0 0'+lines[idx][len('6 4 0 0 0 0'):]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)

def mseg_attribute_update2(input_dir, output_dir):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list, desc='risk to l2'):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                parts = list(map(float, parts))
                for i in range(2, 6):
                    parts[i] = int(parts[i]>0)
                parts[0] = int(parts[0])
                parts[1] = int(parts[1])
                parts = list(map(str, parts))
                lines[idx] = ' '.join(parts)+'\n'
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        shutil.copy(input_image_path, output_image_path)

def compute_polygon_area(uv_list):
    u_coords = uv_list[::2]  # 偶数索引是 u
    v_coords = uv_list[1::2]  # 奇数索引是 v

    # 计算 min/max
    umin, umax = min(u_coords), max(u_coords)
    vmin, vmax = min(v_coords), max(v_coords)

    # 计算面积
    width = umax - umin
    height = vmax - vmin
    area = width * height

    return area

def mseg_line2boxarea(line):
    parts = line.strip().split(' ')
    att_len = int(parts[1])
    uvs = parts[2+att_len:]
    uvs = list(map(float, uvs))
    area = compute_polygon_area(uvs)*960*960
    return area

def mseg_get_large_object(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    mseg_get_large_object_gt(input_label_dir, output_label_dir)
    if cp_img:
        shutil.copytree(input_image_dir, output_image_dir, dirs_exist_ok=True)

def mseg_get_large_object_gt(input_label_dir, output_label_dir):
    label_list = os.listdir(input_label_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        new_lines = []
        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if mseg_line2boxarea(line) > 96*96:
                    new_lines.append(line)
        with open(output_label_path, 'w') as f:
            f.writelines(new_lines)

def seg_class_update_gt(input_dir, output_dir):
    label_list = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for label_name in tqdm(label_list):
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
def seg_class_update(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def seg_class_remove(input_dir, output_dir, remove_class, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                if line[0] == str(remove_class):
                    continue
                else:
                    new_lines.append(lines[idx])
        with open(output_label_path, 'w') as f:
            f.writelines(new_lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def seg_filter_small(input_dir, output_dir, cp_img = True, threshold=0.01, class_list=[2,4,6,7], with_attribute=True):
    input_image_dir = osp.join(input_dir, 'images')
    input_label_dir = osp.join(input_dir, 'labels')
    output_image_dir = osp.join(output_dir, 'images')
    output_label_dir = osp.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_list = os.listdir(input_image_dir)
    areas_poly, areas_bbox = [], []
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        area_poly, area_bbox = filter_yolo_segmentation(input_label_path, output_label_path, threshold=threshold, with_attribute=with_attribute, class_list=class_list)
        areas_poly += area_poly
        areas_bbox += area_bbox

        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

    analyze_area_distribution(areas_poly, os.path.join(output_dir, 'areas_poly.png'))
    analyze_area_distribution(areas_bbox, os.path.join(output_dir, 'areas_bbox.png'))


def seg_filter_small_gt(input_label_dir, output_label_dir, threshold=0.01, class_list=[2, 4, 6, 7], with_attribute=True):
    os.makedirs(output_label_dir, exist_ok=True)

    label_list = os.listdir(input_label_dir)
    areas_poly, areas_bbox = [], []
    for label_name in tqdm(label_list):

        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)


        area_poly, area_bbox = filter_yolo_segmentation(input_label_path, output_label_path, threshold=threshold,
                                                        with_attribute=with_attribute, class_list=class_list)
        areas_poly += area_poly
        areas_bbox += area_bbox


def seg_filter_and_remove(input_dir, output_dir, remove_class, cp_img=True, threshold=0.01, class_list=[2,4,6,7], with_attribute=True):
    tp_dir = output_dir+'_tp'
    seg_class_remove(input_dir, tp_dir, remove_class, cp_img=cp_img)
    seg_filter_small(tp_dir, output_dir, cp_img=cp_img, threshold=threshold, class_list=class_list, with_attribute=with_attribute)


def analyze_area_distribution(area_list, save_path=None, bins='auto', plot=True):
    """
    分析面积分布并生成统计报告

    参数:
        area_list (list): 面积数值列表（单位：像素面积）
        bins (int/str): 直方图分箱策略，默认自动选择
        plot (bool): 是否显示可视化图表

    返回:
        dict: 包含统计摘要和分箱数据的字典
    """
    # 输入验证
    if not isinstance(area_list, (list, np.ndarray)):
        raise ValueError("输入必须是列表或numpy数组")
    if len(area_list) == 0:
        raise ValueError("输入列表不能为空")

    # 转换为numpy数组
    areas = np.array(area_list)

    # 基础统计量
    stats_result = {
        'total': len(areas),
        'mean': np.mean(areas),
        'median': np.median(areas),
        'std': np.std(areas),
        'min': np.min(areas),
        'max': np.max(areas),
        'sum': np.sum(areas),
        'variance': np.var(areas),
        'skewness': stats.skew(areas) if len(areas) > 2 else 0,
        'kurtosis': stats.kurtosis(areas) if len(areas) > 3 else 0
    }

    # 分箱统计
    hist, bin_edges = np.histogram(areas, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_counts = hist.astype(int).tolist()

    distribution = {
        'bins': bin_edges.tolist(),
        'counts': bin_counts,
        'centers': bin_centers.tolist()
    }

    # 可视化
    if plot:
        plt.figure(figsize=(12, 6))

        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(areas, bins=bins, color='skyblue', edgecolor='black')
        plt.title('面积分布直方图')
        plt.xlabel('面积 (像素)')
        plt.ylabel('频数')
        plt.grid(axis='y', alpha=0.75)

        # 累积分布曲线
        plt.subplot(1, 2, 2)
        sorted_areas = np.sort(areas)
        plt.plot(sorted_areas, np.arange(1, len(sorted_areas) + 1) / len(sorted_areas))
        plt.title('累积分布函数 (CDF)')
        plt.xlabel('面积 (像素)')
        plt.ylabel('累积比例')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()
        if save_path is not None:
            plt.savefig(save_path)
    analysis_result = {
        'statistics': stats_result,
        'distribution': distribution
    }
    print("=== 面积分布统计 ===")
    print(f"样本总数: {analysis_result['statistics']['total']}")
    print(f"平均面积: {analysis_result['statistics']['mean']:.1f}")
    print(f"中位面积: {analysis_result['statistics']['median']:.1f}")
    print(f"标准差: {analysis_result['statistics']['std']:.1f}")
    print(f"面积范围: {analysis_result['statistics']['min']:.1f} - {analysis_result['statistics']['max']:.1f}")
    print(f"偏度: {analysis_result['statistics']['skewness']:.2f}")
    print(f"峰度: {analysis_result['statistics']['kurtosis']:.2f}")

def filter_yolo_segmentation(input_file, output_file, threshold, with_attribute=False, class_list=[]):
    def calculate_polygon_area(points):
        """使用鞋带公式计算多边形面积"""
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0

    def calculate_bbox_area(points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        area_axis_aligned = width * height
        return area_axis_aligned
    areas_poly, areas_bbox = [], []
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split()
            if not parts:
                continue

            # 解析类别和坐标点
            class_id = int(parts[0])
            if class_id not in class_list:
                f_out.write(line + '\n')
                continue
            if with_attribute:
                att_len= int(parts[1])
                coords = list(map(float, parts[2+att_len:]))
            else:
                coords = list(map(float, parts[1:]))


            # 转换归一化坐标为实际坐标
            normalized_points = [(coords[i], coords[i + 1])
                                 for i in range(0, len(coords), 2)]

            # 计算面积并过滤
            area_poly = calculate_polygon_area(normalized_points)
            area_bbox = calculate_bbox_area(normalized_points)
            areas_poly.append(area_poly)
            areas_bbox.append(area_bbox)
            if area_bbox >= threshold:
                f_out.write(line + '\n')
    return areas_poly, areas_bbox

def random_select(data_dir, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True, suffix=''):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    file_list = os.listdir(image_dir)
    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(image_dir)
    if full_path:
        file_list = [os.path.join(image_dir, filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    df_train.to_csv(os.path.join(save_dir, f'train{suffix}.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, f'val{suffix}.txt'), header=None, index=None)
    df_all.to_csv(os.path.join(save_dir, 'all.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, f'train{suffix}.txt'),
                                           len(val_list), os.path.join(save_dir, f'val{suffix}.txt')))


def random_select_exclude(data_dir, exclude_data_dir, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True, suffix=''):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    exclude_image_dir = os.path.join(exclude_data_dir, 'images')
    file_list = os.listdir(image_dir)
    exclude_image_list = os.listdir(exclude_image_dir)
    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(image_dir)
    src_num = len(file_list)
    file_list = [filename for filename in file_list if filename not in exclude_image_list]
    dst_num = len(file_list)
    print(f'{src_num} --> {dst_num}, exclude {src_num-dst_num} in {len(file_list)}')
    if full_path:
        file_list = [os.path.join(image_dir, filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    df_train.to_csv(os.path.join(save_dir, f'train{suffix}.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, f'val{suffix}.txt'), header=None, index=None)
    df_all.to_csv(os.path.join(save_dir, 'all.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, f'train{suffix}.txt'),
                                           len(val_list), os.path.join(save_dir, f'val{suffix}.txt')))

def random_kfold(img_dir, k, label_dir=None, save_dir=None, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]

    np.random.seed(random_seed)
    np.random.shuffle(file_list)

    total_images = len(file_list)
    fold_size = [total_images // k] * k
    for i in range(total_images % k):
        fold_size[i] += 1

    start_idx = 0
    for fold in range(k):
        end_idx = start_idx + fold_size[fold]
        val_ids = list(range(start_idx, end_idx))
        train_ids = [i for i in range(total_images) if i not in val_ids]
        train_list = [file_list[i] for i in train_ids]
        val_list = [file_list[i] for i in val_ids]

        df_train = pd.DataFrame({'filename': train_list})
        df_val = pd.DataFrame({'filename': val_list})
        train_path = os.path.join(save_dir, f'train_{fold}.txt')
        val_path = os.path.join(save_dir, f'val_{fold}.txt')
        df_train.to_csv(train_path, header=None, index=None)
        df_val.to_csv(val_path, header=None, index=None)
        print('%d save to %s,\n%d save to %s!' % (len(train_list), train_path,
                                                  len(val_list), val_path))

def ref_split(ref_path, img_dir, label_dir=None, save_dir=None, full_path=True, add_suffix='_ref'):
    file_list = os.listdir(img_dir)
    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(img_dir)


    df = pd.read_csv(ref_path, header=None, index_col=None, names=['path'])
    ref_list = [Path(file_path).stem for file_path in df['path'].tolist()]

    train_list = [file_path for file_path in file_list if Path(file_path).stem not in ref_list]
    val_list = [file_path for file_path in file_list if Path(file_path).stem in ref_list]
    if full_path:
        val_list = [os.path.join(img_dir, filename) for filename in val_list]
        train_list = [os.path.join(img_dir, filename) for filename in train_list]
    else:
        val_list = [os.path.join('images', filename) for filename in val_list]
        train_list = [os.path.join('images', filename) for filename in train_list]
    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    train_path = os.path.join(save_dir, f'train{add_suffix}.txt')
    val_path = os.path.join(save_dir, f'val{add_suffix}.txt')
    all_path = os.path.join(save_dir, 'all.txt')
    df_train.to_csv(train_path, header=None, index=None)
    df_val.to_csv(val_path, header=None, index=None)
    df_all.to_csv(all_path, header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), train_path, len(val_list), val_path))

def copy_split(input_path, output_path, abs_path=False):
    df = pd.read_csv(input_path, header=None, index_col=None, names=['path'])
    path_list = df['path'].tolist()
    if abs_path:
        output_dir = os.path.join(os.path.dirname(output_path), 'images')
    else:
        output_dir = 'images'
    output_path_list = [os.path.join(output_dir, Path(file_path).name) for file_path in path_list]
    df_output = pd.DataFrame({'path': output_path_list})
    df_output.to_csv(output_path, header=None, index=None, encoding='utf-8')
    print(f'copy {os.path.basename(input_path)}, {len(df_output)} records')

def split_add(input_path, ref_dir, output_path):
    df = pd.read_csv(input_path, header=None, index_col=None, names=['path'])
    path_list = df['path'].tolist()
    base_dir = os.path.dirname(path_list[0])

    ref_list = os.listdir(ref_dir)
    ref_list_add = [os.path.join(base_dir, ref_name) for ref_name in ref_list if not ref_name.endswith('.json')]

    output_path_list = path_list + ref_list_add
    df_output = pd.DataFrame({'path': output_path_list})
    df_output.to_csv(output_path, header=None, index=None, encoding='utf-8')
    print(f'src {len(path_list)}, add {len(ref_list_add)}, total {len(output_path_list)} records')

def split_txt_merge(input_dir1, input_dir2, output_dir, suffix):
    input_train_path1 = os.path.join(input_dir1, f'train{suffix}.txt')
    input_train_path2 = os.path.join(input_dir2, f'train{suffix}.txt')
    output_train_path = os.path.join(output_dir, f'train{suffix}.txt')
    input_val_path1 = os.path.join(input_dir1, f'val{suffix}.txt')
    input_val_path2 = os.path.join(input_dir2, f'val{suffix}.txt')
    output_val_path = os.path.join(output_dir, f'val{suffix}.txt')
    df_input_train1 = pd.read_csv(input_train_path1, names=['file_name'], header=None, index_col=False)
    df_input_train1['file_name'] = df_input_train1['file_name'].str.replace(input_dir1, output_dir)
    df_input_train2 = pd.read_csv(input_train_path2, names=['file_name'], header=None, index_col=False)
    df_input_train2['file_name'] = df_input_train2['file_name'].str.replace(input_dir2, output_dir)
    df_output_train = pd.concat([df_input_train1, df_input_train2])
    df_output_train.to_csv(output_train_path, index=False, header=False)
    df_input_val1 = pd.read_csv(input_val_path1, names=['file_name'], header=None, index_col=False)
    df_input_val1['file_name'] = df_input_val1['file_name'].str.replace(input_dir1, output_dir)
    df_input_val2 = pd.read_csv(input_val_path2, names=['file_name'], header=None, index_col=False)
    df_input_val2['file_name'] = df_input_val2['file_name'].str.replace(input_dir2, output_dir)
    df_output_val = pd.concat([df_input_val1, df_input_val2])
    df_output_val.to_csv(output_val_path, index=False, header=False)
    print(f'merge {input_train_path1} + {input_train_path2} -> {output_train_path}')

def select_val(input_dir, val_txt='val.txt'):
    pass
    print(f'select {input_dir} by {val_txt}')
    val_txt_path = os.path.join(input_dir, val_txt)
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    val_dir = os.path.join(input_dir, val_txt_path.replace('.txt', ''))
    val_image_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    if os.path.exists(val_label_dir):
        shutil.rmtree(val_label_dir)
    if os.path.exists(val_image_dir):
        shutil.rmtree(val_image_dir)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    df = pd.read_csv(val_txt_path, header=None, index_col=False, names=['image_name'])
    img_list = df['image_name'].to_list()
    for image_path in tqdm(img_list):
        image_name = Path(image_path).name
        txt_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(image_dir, image_name)
        ouput_image_path = os.path.join(val_image_dir, image_name)
        input_label_path = os.path.join(label_dir, txt_name)
        output_label_path = os.path.join(val_label_dir, txt_name)
        shutil.copy(input_image_path, ouput_image_path)
        shutil.copy(input_label_path, output_label_path)

    print('select finish!\n')


def data_merge(input_dir1, input_dir2, output_dir, cp_split=True, suffix=''):
    print(f'merging {input_dir1} + {input_dir2} --> {output_dir}...')
    data_copy(input_dir1, output_dir)
    data_copy(input_dir2, output_dir)
    if cp_split:
        split_txt_merge(input_dir1, input_dir2, output_dir, suffix)
    print(f'merging {input_dir1} + {input_dir2} --> {output_dir} finished!')

def data_copy(input_dir, output_dir):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list, desc='data copying:'):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        shutil.copy(input_image_path, output_image_path)
        shutil.copy(input_label_path, output_label_path)

def poly2xywh(mask):
    mask = np.array([mask[::2], mask[1::2]])
    x_min,y_min = np.min(mask, axis=1)
    x_max,y_max = np.max(mask, axis=1)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]


def get_attributes(attribute_path):
    with open(attribute_path, 'r') as file:
        attribute_dict = yaml.safe_load(file)['attributes']
    attribute_keys = list(attribute_dict.keys())
    return attribute_keys

def df_xywh_to_xyxy(df):
    df = df.copy()
    df["x1"] = df["x"] - df["w"] / 2
    df["y1"] = df["y"] - df["h"] / 2
    df["x2"] = df["x"] + df["w"] / 2
    df["y2"] = df["y"] + df["h"] / 2
    return df

def get_yolo_label_df(gt_path, mdet=False, attributes=None, with_track_id=False, with_object_id=False, with_conf=False,
                      conf_threshold=0.001, defect_conf_threshold=None):
    if mdet:
        assert attributes is not None, 'attribute_path must be provided, which is "%s"' % attributes
        if isinstance(attributes, str):
            attribute_keys = get_attributes(attributes)
        elif isinstance(attributes, list):
            attribute_keys = attributes
        names = ['category'] + ['attribute_len'] + attribute_keys + ['x', 'y', 'w', 'h', 'image']
    else:
        names = ['category', 'x', 'y', 'w', 'h', 'image_name']
    if with_track_id:
        names = names + ['track_id']
    if with_object_id:
        names = names + ['id']
    if with_conf:
        names = names + ['conf']


    df = pd.DataFrame(None, columns=names)
    with open(gt_path, 'r') as f:
        data = f.readlines()
        for id_line, line in enumerate(data):
            parts = line.strip().split(' ')
            if with_conf:
                conf = float(parts[-1])
                parts = parts[:-1]
            category = int(parts[0])
            image_name = Path(gt_path).name
            if mdet:
                att_len = int(parts[1])
                atts = list(map(float, parts[2:2 + att_len]))
                if with_track_id:
                    track_id = int(parts[-1])
                    polygons = list(map(float, parts[2 + att_len:-1]))
                    if len(polygons) < 4:
                        continue
                    xywh = poly2xywh(polygons)
                    info = [category, att_len] + atts + xywh + [image_name, track_id]
                else:
                    polygons = list(map(float, parts[2 + att_len:]))
                    if len(polygons) < 4:
                        continue
                    xywh = poly2xywh(polygons)
                    info = [category, att_len] + atts + xywh + [image_name]
            else:
                polygons = list(map(float, parts[1:]))
                xywh = poly2xywh(polygons)
                info = [category] + xywh + [image_name]
            if with_object_id:
                info += [id_line]
            if with_conf:
                info += [conf]
            df.loc[len(df)] = info

    if mdet:
        df['defect'] = (df[attribute_keys] > 0).any(axis=1)
        attribute_keys_noc = [a for a in attribute_keys if a != 'corrosion']
        df['defect_no_c'] = (df[attribute_keys_noc] > 0).any(axis=1)
    if with_conf:
        if defect_conf_threshold is None:
            df = df[(df['conf'] >= conf_threshold)]
        else:
            # df = df[~(
            #     ((df['defect_no_c'] == True) & (df['conf'] < defect_conf_threshold)) |
            #     ((df['defect_no_c'] == False) & (df['conf'] < conf_threshold))
            # )]
            df = df[~(
                ((df['defect'] == True) & (df['conf'] < defect_conf_threshold)) |
                ((df['defect'] == False) & (df['conf'] < conf_threshold))
            )]
    df = df_xywh_to_xyxy(df)
    return df

def data_check(label_dir, attribute_path=None, mdet=False, check_item='category'):
    if 'labels' not in label_dir:
        label_dir = os.path.join(label_dir, 'labels')
    attributes = get_attributes(attribute_path) if attribute_path is not None else None
    label_list = os.listdir(label_dir)
    check_result = []
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        df_label = get_yolo_label_df(label_path, mdet=mdet, attributes=attributes)
        if check_item == 'category':
            for idx, row in df_label.iterrows():
                cat_id = row['category']
                if cat_id == 0:
                    print(label_name)
                    check_result.append(label_name)
        elif check_item == 'attribute':
            cond1 = df_label['category']==6
            cond2 = ((df_label['abandonment']==1.0) | (df_label['broken']==1.0) | (df_label['corrosion']==1.0) | (df_label['deformation']==1.0))
            cond = cond1 & cond2
            df_condition = df_label[cond]
            count = df_condition.shape[0]
            if count>0:
                print(label_name)
                check_result.append(label_name)
        else:
            print(f'check item: {check_item}, not support')
            break
    return check_result

def copy_dataset(input_dir, output_dir, class_file=None, attribute_file=None):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        shutil.copy(input_image_path, output_image_path)
        shutil.copy(input_label_path, output_label_path)

    if class_file is not None:
        output_class_file = os.path.join(output_dir, Path(class_file).name)
        shutil.copy(class_file, output_class_file)
    if attribute_file is not None:
        output_attribute_file = os.path.join(output_dir, Path(attribute_file).name)
        shutil.copy(attribute_file, output_attribute_file)


def new_psdata_add_pipline(input_ps_mseg_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir):
    pass
    input_ps_mseg_c6_dir = input_ps_mseg_dir + '_c6'
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_large_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_large')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')

    # get ps mseg c6 dataset
    mseg_class_update(
        input_ps_mseg_dir,
        input_ps_mseg_c6_dir
    )
    random_select(input_ps_mseg_c6_dir)
    shutil.copy(ATT_FILE, os.path.join(input_ps_mseg_c6_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C6_FILE, os.path.join(input_ps_mseg_c6_dir, 'class.txt'))

    # get fused mseg c6 dataset
    data_merge(
        input_ps_mseg_c6_dir,
        src_fused_mseg_c6_dir,
        dst_fused_mseg_c6_dir,
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c6_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_mseg_c6_dir, 'class.txt'))

    # get fused mseg c5 dataset
    mseg_class_update2(
        dst_fused_mseg_c6_dir,
        dst_fused_mseg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))


    # get fused mseg c5 l2 dataset
    mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))

    # get fused seg c6 dataset
    mseg2seg(
        dst_fused_mseg_c6_dir,
        dst_fused_seg_c6_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c6_dir, 'images'),
        os.path.join(dst_fused_seg_c6_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))

    # get fused seg c5 dataset
    mseg2seg(
        dst_fused_mseg_c5_dir,
        dst_fused_seg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))

    mseg_get_large_object(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_large_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'labels'),
        add_suffix=''
    )



def new_psdata_add_pipline2(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir):
    pass
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_large_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_large')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')


    random_select(input_ps_mseg_c6_dir)
    random_select(input_ps_mseg_c6_dir, train_ratio=0.8, suffix='_80p')
    # get fused mseg c6 dataset
    data_merge(
        input_ps_mseg_c6_dir,
        src_fused_mseg_c6_dir,
        dst_fused_mseg_c6_dir,
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c6_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_mseg_c6_dir, 'class.txt'))

    # get fused mseg c5 dataset
    mseg_class_update2(
        dst_fused_mseg_c6_dir,
        dst_fused_mseg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val_80p.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))


    # get fused mseg c5 l2 dataset
    mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val_80p.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))

    # get fused seg c6 dataset
    mseg2seg(
        dst_fused_mseg_c6_dir,
        dst_fused_seg_c6_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c6_dir, 'images'),
        os.path.join(dst_fused_seg_c6_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))

    # get fused seg c5 dataset
    mseg2seg(
        dst_fused_mseg_c5_dir,
        dst_fused_seg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val_80p.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))

    mseg_get_large_object(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_large_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'labels'),
        add_suffix=''
    )


def new_psdata_add_pipline3(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir):
    pass
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_large_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_large')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')


    random_select(input_ps_mseg_c6_dir)
    random_select(input_ps_mseg_c6_dir, train_ratio=1, suffix='_80p')
    # get fused mseg c6 dataset
    data_merge(
        input_ps_mseg_c6_dir,
        src_fused_mseg_c6_dir,
        dst_fused_mseg_c6_dir,
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c6_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_mseg_c6_dir, 'class.txt'))

    # get fused mseg c5 dataset
    mseg_class_update2(
        dst_fused_mseg_c6_dir,
        dst_fused_mseg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val_80p.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))


    # get fused mseg c5 l2 dataset
    mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val_80p.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))

    # get fused seg c6 dataset
    mseg2seg(
        dst_fused_mseg_c6_dir,
        dst_fused_seg_c6_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c6_dir, 'images'),
        os.path.join(dst_fused_seg_c6_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))

    # get fused seg c5 dataset
    mseg2seg(
        dst_fused_mseg_c5_dir,
        dst_fused_seg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix=''
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val_80p.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix='_80p'
    )
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))

    mseg_get_large_object(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_large_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'labels'),
        add_suffix=''
    )

def new_psdata_add_pipline2_80(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir):
    pass
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_large_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_large')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')


    random_select(input_ps_mseg_c6_dir)

    # get fused mseg c6 dataset
    data_merge(
        input_ps_mseg_c6_dir,
        src_fused_mseg_c6_dir,
        dst_fused_mseg_c6_dir,
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c6_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_mseg_c6_dir, 'class.txt'))

    # get fused mseg c5 dataset
    mseg_class_update2(
        dst_fused_mseg_c6_dir,
        dst_fused_mseg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))


    # get fused mseg c5 l2 dataset
    mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))

    # get fused seg c6 dataset
    mseg2seg(
        dst_fused_mseg_c6_dir,
        dst_fused_seg_c6_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c6_dir, 'images'),
        os.path.join(dst_fused_seg_c6_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))

    # get fused seg c5 dataset
    mseg2seg(
        dst_fused_mseg_c5_dir,
        dst_fused_seg_c5_dir,
    )
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix=''
    )
    shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))

    mseg_get_large_object(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_large_dir)
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, 'val.txt'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_large_dir, 'labels'),
        add_suffix=''
    )


def psdata_add_piplines(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir, add_train_ratio=1, selected_suffix_list=[], copy=True):
    for idx, selected_suffix in enumerate(selected_suffix_list):
        if copy and idx==0:
            pass
        else:
            copy = False
        psdata_add_pipline(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir, add_train_ratio, selected_suffix, copy)

def psdata_add_pipline(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir, add_train_ratio=1, selected_suffix='', copy=True):
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')

    # split current data
    random_select(input_ps_mseg_c6_dir, train_ratio=add_train_ratio, suffix=selected_suffix)


    # get fused mseg c6 dataset
    if copy:
        data_merge(
            input_ps_mseg_c6_dir,
            src_fused_mseg_c6_dir,
            dst_fused_mseg_c6_dir,
            cp_split=True,
            suffix=selected_suffix,
        )
        shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c6_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_mseg_c6_dir, 'class.txt'))
    else:
        split_txt_merge(input_ps_mseg_c6_dir, src_fused_mseg_c6_dir, dst_fused_mseg_c6_dir, suffix=selected_suffix,)

    # get fused mseg c5 dataset
    if copy:
        mseg_class_update2(
            dst_fused_mseg_c6_dir,
            dst_fused_mseg_c5_dir,
        )
        shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=selected_suffix
    )


    # get fused mseg c5 l2 dataset
    if copy:
        mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
        shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=selected_suffix
    )



    # get fused seg c6 dataset
    if copy:
        mseg2seg(
            dst_fused_mseg_c6_dir,
            dst_fused_seg_c6_dir,
        )
        shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_seg_c6_dir, 'images'),
        os.path.join(dst_fused_seg_c6_dir, 'labels'),
        add_suffix=selected_suffix
    )


    # get fused seg c5 dataset
    if copy:
        mseg2seg(
            dst_fused_mseg_c5_dir,
            dst_fused_seg_c5_dir,
        )
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_seg_c5_dir, 'images'),
        os.path.join(dst_fused_seg_c5_dir, 'labels'),
        add_suffix=selected_suffix
    )


def data_tf_piplines(dst_fused_mseg_c6_dir, train_ratio_list=[], selected_suffix_list=[], copy=True):
    for idx in range(len(selected_suffix_list)):
        train_ratio = train_ratio_list[idx]
        selected_suffix = selected_suffix_list[idx]
        if copy and idx==0:
            pass
        else:
            copy = False
        data_tf_pipline(dst_fused_mseg_c6_dir, train_ratio=train_ratio, selected_suffix=selected_suffix, copy=True)

def data_tf_pipline(dst_fused_mseg_c6_dir, train_ratio=1, selected_suffix='', copy=True, split_mseg_c6=False):
    dst_fused_seg_c6_dir = dst_fused_mseg_c6_dir.replace('_mseg', '_seg')
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('_mseg', '_seg')

    # split current data
    if split_mseg_c6:
        random_select(dst_fused_mseg_c6_dir, train_ratio=train_ratio, suffix=selected_suffix)


    # get fused mseg c5 dataset
    if copy:
        mseg_class_update2(
            dst_fused_mseg_c6_dir,
            dst_fused_mseg_c5_dir,
        )
        shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c6_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_mseg_c5_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_dir, 'labels'),
        add_suffix=selected_suffix
    )


    # get fused mseg c5 l2 dataset
    if copy:
        mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
        shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))
    ref_split(
        os.path.join(dst_fused_mseg_c5_dir, f'val{selected_suffix}.txt'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'images'),
        os.path.join(dst_fused_mseg_c5_l2_dir, 'labels'),
        add_suffix=selected_suffix
    )


    #
    # # get fused seg c6 dataset
    # if copy:
    #     mseg2seg(
    #         dst_fused_mseg_c6_dir,
    #         dst_fused_seg_c6_dir,
    #     )
    #     shutil.copy(CLASS_C6_FILE, os.path.join(dst_fused_seg_c6_dir, 'class.txt'))
    # ref_split(
    #     os.path.join(dst_fused_mseg_c6_dir, f'val{selected_suffix}.txt'),
    #     os.path.join(dst_fused_seg_c6_dir, 'images'),
    #     os.path.join(dst_fused_seg_c6_dir, 'labels'),
    #     add_suffix=selected_suffix
    # )


    # # get fused seg c5 dataset
    # if copy:
    #     mseg2seg(
    #         dst_fused_mseg_c5_dir,
    #         dst_fused_seg_c5_dir,
    #     )
    #     shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))
    # ref_split(
    #     os.path.join(dst_fused_mseg_c5_dir, f'val{selected_suffix}.txt'),
    #     os.path.join(dst_fused_seg_c5_dir, 'images'),
    #     os.path.join(dst_fused_seg_c5_dir, 'labels'),
    #     add_suffix=selected_suffix
    # )


def data_tf_pipline_new(dst_fused_mseg_c6_dir,  copy_list=['mseg_c5', 'mseg_c5_l2', 'seg_c5']):
    dst_fused_mseg_c5_dir = dst_fused_mseg_c6_dir.replace('_c6', '_c5')
    dst_fused_mseg_c5_l2_dir = dst_fused_mseg_c5_dir.replace('_c5', '_c5_l2')
    dst_fused_seg_c5_dir = dst_fused_mseg_c5_dir.replace('mseg', 'seg')

    # get fused mseg c5 dataset
    if 'mseg_c5' in copy_list or 'mseg_c5_l2' in copy_list or 'seg_c5' in copy_list:
        mseg_class_update2(
            dst_fused_mseg_c6_dir,
            dst_fused_mseg_c5_dir,
        )
        shutil.copy(ATT_FILE, os.path.join(dst_fused_mseg_c5_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_dir, 'class.txt'))

    # get fused mseg c5 l2 dataset
    if 'mseg_c5_l2' in copy_list:
        mseg_attribute_update2(dst_fused_mseg_c5_dir, dst_fused_mseg_c5_l2_dir)
        shutil.copy(ATT_L2_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'attribute.yaml'))
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_mseg_c5_l2_dir, 'class.txt'))

    # get fused seg c5 dataset
    if 'seg_c5' in copy_list:
        mseg2seg(
            dst_fused_mseg_c5_dir,
            dst_fused_seg_c5_dir,
        )
        shutil.copy(CLASS_C5_FILE, os.path.join(dst_fused_seg_c5_dir, 'class.txt'))


def att_check(input_dir):
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list):
        label_path = os.path.join(input_dir, label_name)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[2] == '0':
                    lines[idx] = lines[idx][0:2] + '4' + lines[idx][3:]
                else:
                    continue
        with open(label_path, 'w') as f:
            f.writelines(lines)

def img_label_cross_check(file_list, label_list):
    label_list = [Path(label_name).stem for label_name in label_list]
    file_list_check = []
    for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
        name = Path(img_name).stem
        if name in label_list:
            file_list_check.append(img_name)
    file_list = file_list_check
    return file_list


def split_rmbd(input_dir, bd_dir, train_ratio=0.8, random_seed=1010, add_suffix=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    bd_label_dir = os.path.join(bd_dir, 'labels')
    bd_image_dir = os.path.join(bd_dir, 'images')

    file_list = os.listdir(input_image_dir)
    label_list = os.listdir(input_label_dir)
    bd_file_list = os.listdir(bd_image_dir)
    bd_label_list = os.listdir(bd_label_dir)


    file_list = img_label_cross_check(file_list, label_list)
    bd_file_list = img_label_cross_check(bd_file_list, bd_label_list)
    file_list = [x for x in file_list if x not in bd_file_list]

    if add_suffix:
        add_suffix_str = f'_{int(train_ratio*100)}p'
    else:
        add_suffix_str = ''

    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num] + bd_file_list
    val_list = file_list[train_num:]
    train_list = [os.path.join(input_image_dir, file_name) for file_name in train_list]
    val_list = [os.path.join(input_image_dir, file_name) for file_name in val_list]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    train_path = os.path.join(input_dir, f'train{add_suffix_str}.txt')
    val_path = os.path.join(input_dir, f'val{add_suffix_str}.txt')
    all_path = os.path.join(input_dir, 'all.txt')
    df_train.to_csv(train_path, header=None, index=None)
    df_val.to_csv(val_path, header=None, index=None)
    df_all.to_csv(all_path, header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), train_path, len(val_list), val_path))


def ref_k_fold(input_dir, k=5):
    val_path = os.path.join(input_dir, "val.txt")
    images_dir = os.path.join(input_dir, "images")
    # 1. 读取 val.txt 中的图像路径（20% 数据）
    with open(val_path, "r") as f:
        val_images = f.read().splitlines()  # 每行是图像路径，如 "images/img1.jpg"

    # 2. 获取所有图像路径（排除 val.txt 中的图像）
    all_images = []
    for img_file in os.listdir(images_dir):
        img_path = os.path.join("images", img_file)  # "images/img1.jpg"
        if img_path not in val_images:  # 排除 val.txt 中的图像
            all_images.append(img_path)

    # 3. 随机打乱剩余数据
    np.random.shuffle(all_images)

    # 4. 划分 4-Fold（每个 Fold 占 20% 原始数据）
    kf = KFold(n_splits=k, shuffle=True, random_state=1010)
    folds = list(kf.split(all_images))  # 返回 (train_idx, test_idx) 对

    # 5. 生成 4 组 train/test 文件
    for i, (train_idx, test_idx) in enumerate(folds):
        # 训练集：3 个 Fold（60% 原始数据）
        train_images = [all_images[idx] for idx in train_idx]
        # 测试集：1 个 Fold（20% 原始数据）
        test_images = [all_images[idx] for idx in test_idx]

        # 写入 train_fold_{i}.txt 和 test_fold_{i}.txt
        with open(os.path.join(input_dir, f"train_fold_{i}.txt"), "w") as f:
            f.write("\n".join(train_images))
        with open(os.path.join(input_dir, f"val_fold_{i}.txt"), "w") as f:
            f.write("\n".join(test_images))

        print(f"Fold {i}:")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val: {len(test_images)} images")


def count_files(input_dir):
    total = 0
    for _,_, files in os.walk(input_dir):
        total += len(files)
    return total

def delete_matching_files(input_dir1, input_dir2, dry_run=True):
    before_opt = count_files(input_dir1)

    stems_ref = set(Path(f).stem for f in os.listdir(input_dir2))

    delected_count = 0

    for root, dirs, files in os.walk(input_dir1):
        for file in files:
            file_path = os.path.join(root, file)
            name_stem = Path(file).stem

            base_stem, obj_id = name_stem.rsplit('_', 1)
            if base_stem in stems_ref:
                if dry_run:
                    print(f'[PRE] delete: {name_stem}')
                else:
                    print(f'[INFO] delete: {name_stem}')
                    os.remove(file_path)
                delected_count += 1
    after_opt = count_files(input_dir1)

    print(f'before: {before_opt}, after: {after_opt}, delected: {delected_count}')

def get_stem2name(input_dir):
    name_list = os.listdir(input_dir)
    stem2name_dict = {Path(name).stem:name for name in name_list}
    return stem2name_dict

def copy_ref_list(input_dir, output_dir, ref_list):
    os.makedirs(output_dir, exist_ok=True)
    input_stem2name = get_stem2name(input_dir)
    for ref_name in tqdm(ref_list):
        input_name = input_stem2name[Path(ref_name).stem]
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, input_name)
        shutil.copy(input_path, output_path)

def copy_ref_dir(input_dir, output_dir, ref_dir):
    ref_list = os.listdir(ref_dir)
    copy_ref_list(input_dir, output_dir, ref_list)

def copy_ref_xlsx(input_dir, output_dir, ref_path, column='file_name'):
    df = pd.read_excel(ref_path, sheet_name='Sheet1')
    ref_list = df[column].to_list()
    copy_ref_list(input_dir, output_dir, ref_list)

def copy_ref_csv(input_dir, output_dir, ref_path):
    df = pd.read_csv(ref_path, names=['path'])
    ref_list = df['path'].to_list()
    copy_ref_list(input_dir, output_dir, ref_list)

def copy_exclude_list(input_dir, output_dir, exclude_list):
    os.makedirs(output_dir, exist_ok=True)
    input_list = os.listdir(input_dir)
    exclude_stem_list = [Path(exclude_name).stem for exclude_name in exclude_list]
    for input_name in tqdm(input_list):
        input_stem = Path(input_name).stem
        if input_stem not in exclude_stem_list:
            input_path = os.path.join(input_dir, input_name)
            output_path = os.path.join(output_dir, input_name)
            shutil.copy(input_path, output_path)

def copy_exclude_dir(input_dir, output_dir, exclude_dir):
    exclude_list = os.listdir(exclude_dir)
    copy_exclude_list(input_dir, output_dir, exclude_list)

def copy_exclude_xlsx(input_dir, output_dir, exclude_path, column='file_name'):
    df = pd.read_excel(exclude_path, sheet_name='Sheet1')
    exclude_list = df[column].to_list()
    copy_exclude_list(input_dir, output_dir, exclude_list)

def get_stem2img_dict(img_dir):
    img_list = [img_name for img_name in os.listdir(img_dir) if Path(img_name).suffix.lower() in ['.jpg', '.jpeg', '.png'] ]
    stem_list = [Path(img).stem for img in img_list]
    stem2img_dict = dict(zip(stem_list, img_list))
    return stem2img_dict

def get_stem2name_dict(img_dir):
    img_list = [img_name for img_name in os.listdir(img_dir)]
    stem_list = [Path(img).stem for img in img_list]
    stem2img_dict = dict(zip(stem_list, img_list))
    return stem2img_dict

def find_defect(input_label_dir, output_label_dir, input_image_dir, output_image_dir, defect_list):
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    label_file_list = os.listdir(input_label_dir)
    stem2img_dict = get_stem2img_dict(input_image_dir)
    file_with_defect_list = []
    for label_name in tqdm(label_file_list, desc='find_defect'):
        input_label_path = os.path.join(input_label_dir, label_name)
        df = get_yolo_label_df(input_label_path, mdet=True, attributes=defect_list)
        with_defect = (df[defect_list] > 0).any().any()
        if with_defect:
            label_name_stem = Path(label_name).stem
            image_name = stem2img_dict[label_name_stem]
            input_image_path = os.path.join(input_image_dir, image_name)
            output_label_path = os.path.join(output_label_dir, label_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_label_path, output_label_path)
            shutil.copy(input_image_path, output_image_path)
            file_with_defect_list.append(label_name)
    print(f'find {len(file_with_defect_list)} files with defect')

def att_check(input_dir, output_dir, reorder=False, rm_id=True):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    label_list = os.listdir(input_dir)
    track_list = []
    for label_name in tqdm(label_list, desc='att_check'):
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)
        with open(input_label_path, 'r') as f1, open(output_label_path, 'w') as f2:
            lines = f1.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                risk_list = parts[2:6]
                if reorder:
                    new_risk_list = [risk_list[3], risk_list[1], risk_list[0], risk_list[2]]
                    parts[2:6] = new_risk_list
                if rm_id:
                    if len(parts) % 2 == 1:
                        parts = parts[:-1]
                        track_list.append(label_name)
                new_line = ' '.join(parts) + '\n'
                if line[2] == '0':
                    lines[idx] = lines[idx][0:2] + '4' + lines[idx][3:]
                    count += 1
                new_lines.append(new_line)
            f2.writelines(new_lines)
    print(f'change "0" {count} lines')
    track_list = list(set(track_list))
    print(f'remove {len(track_list)} id')

def copy_all(input_label_dir, output_label_dir, input_image_dir, output_image_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    label_file_list = os.listdir(input_label_dir)
    stem2img_dict = get_stem2img_dict(input_image_dir)
    for label_name in tqdm(label_file_list, desc='copy_all'):
        input_label_path = os.path.join(input_label_dir, label_name)
        label_name_stem = Path(label_name).stem
        image_name = stem2img_dict[label_name_stem]
        input_image_path = os.path.join(input_image_dir, image_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        shutil.copy(input_label_path, output_label_path)
        shutil.copy(input_image_path, output_image_path)


def only_keep_defect_object(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list, desc='only_keep_defect_object'):
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)
        with open(input_label_path, 'r') as f1, open(output_label_path, 'w') as f2:
            lines = f1.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                risk_len = parts[1]
                risks = list(map(int,parts[2:2+int(risk_len)]))
                with_risk = sum(risks) > 0
                if with_risk:
                    new_lines.append(line)
            f2.writelines(new_lines)

def small_judge(parts, filter_small):
    if filter_small is None:
        return False
    # cat = int(parts[0])
    att_len = int(parts[1])
    # atts = list(map(float, parts[2:2 + att_len]))
    polygons = list(map(float, parts[2 + att_len:]))
    if len(polygons)==0:
        return True
    x, y, w, h = poly2xywh(polygons)
    if w>filter_small or h>filter_small:
        return False
    else:
        return True

def remove_conf(input_dir, output_dir, conf_threshold=None, filter_small=None):
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_dir, exist_ok=True)
    input_list = os.listdir(input_dir)
    for label_name in tqdm(input_list, desc='remove_conf'):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(output_dir, label_name)
        with open(input_path, 'r') as f1, open(output_path, 'w') as f2:
            lines = f1.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                if conf_threshold is not None:
                    conf = float(parts[-1])
                    if conf<conf_threshold:
                        continue
                parts = parts[:-1]
                small_obj = small_judge(parts, filter_small)
                if small_obj:
                    continue
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)
            f2.writelines(new_lines)

def copy_all_by_tree(input_dir, output_dir):
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    output_dir_path.mkdir(parents=True, exist_ok=True)

    files = list(input_dir_path.rglob('*'))
    files = [f for f in files if f.is_file()]

    for f in tqdm(files, desc='copy'):
        shutil.copy(f, output_dir_path/f.name)

    # shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    # os.makedirs(output_dir, exist_ok=True)
    # [shutil.copy2(f, Path(output_dir)/f.name) for f in Path(input_dir).rglob('*') if f.is_file()]


def find_empty_file(input_dir, output_path, attributes=None,):
    attributes = get_attributes(attributes)
    input_list = os.listdir(input_dir)

    df_empty = pd.DataFrame(None, columns=['file_path'])
    for label_name in tqdm(input_list, desc='find_empty_file'):
        label_path = os.path.join(input_dir, label_name)
        df = get_yolo_label_df(label_path, mdet=True, attributes=attributes)
        if df is None or df.empty or len(df) == 0:
            df_empty.loc[len(df_empty)] = [label_name]
    df_empty.to_csv(output_path, index=False, header=False)
    print(f'find {len(df_empty)}/{len(input_list)} records')


def copy_dir(input_dir, output_dir):
    input_list = os.listdir(input_dir)
    for input_name in tqdm(input_list, desc='copy_dir'):
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, input_name)
        if not os.path.exists(output_path):
            shutil.copy(input_path, output_path)
        else:
            print(f'copy error, {input_name} exists')
if __name__ == '__main__':
    pass
    # new_psdata_add_pipline(
    #     input_ps_mseg_dir=r'/localnvme/data/billboard/ps_data/psdata_add823_0724_mseg',
    #     src_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data1422_mseg_c6_check0708',
    #     dst_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data2245_mseg_c6_0724'
    # )

    # new_psdata_add_pipline(
    #     input_ps_mseg_dir=r'/localnvme/data/billboard/ps_data/psdata_add997_0730_mseg',
    #     src_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data1422_mseg_c6_check0708',
    #     dst_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data2419_mseg_c6_0730'
    # )

    # att_check(r'/localnvme/data/billboard/ps_data/psdata_add625_0731_mseg_c6/labels')


    # bd_data_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624'
    # split_rmbd(
    #     r'/localnvme/data/billboard/fused_data/data3072_mseg_c5_0809',
    #     bd_data_dir
    # )

    # ref_split(
    #     ref_path=r'/localnvme/data/billboard/fused_data/data3072_seg_c5_0809/val_80p.txt',
    #     img_dir=r'/localnvme/data/billboard/fused_data/data3072_mseg_c5_l2_0809/images',
    #     label_dir=r'/localnvme/data/billboard/fused_data/data3072_mseg_c5_l2_0809/labels',
    #     add_suffix='_80p'
    # )


    # new_psdata_add_pipline2(
    #     input_ps_mseg_c6_dir=r'/localnvme/data/billboard/ps_data/psdata_add625_0731_mseg_c6',
    #     src_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data2419_mseg_c6_0730',
    #     dst_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data3044_mseg_c6_0731'
    # )

    # new_psdata_add_pipline2(
    #     input_ps_mseg_c6_dir=r'/localnvme/data/billboard/ps_data/psdata_add827_0818_mseg_c6',
    #     src_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data3072_mseg_c6_0809',
    #     dst_fused_mseg_c6_dir=r'/localnvme/data/billboard/fused_data/data3899_mseg_c6_0818'
    # )
    # defect_list = ['deformation', 'broken', 'abandonment', 'corrosion']
    # get_yolo_label_df(r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0912/labels/DA5148680_20250812140435100.txt', mdet=True, attributes=defect_list)


    # def get_stem2img_dict(img_dir):
    #     img_list = [img_name for img_name in os.listdir(img_dir)]
    #     stem_list = [Path(img).stem for img in img_list]
    #     stem2img_dict = dict(zip(stem_list, img_list))
    #     return stem2img_dict
    #
    # def find_defect(label_dir, image_dir, defect_list):
    #     defect_file_list = []
    #     label_file_list = os.listdir(label_dir)
    #     stem2img_dict = get_stem2img_dict(image_dir)
    #     for label_name in tqdm(label_file_list):
    #         input_label_path = os.path.join(label_dir, label_name)
    #         df = get_yolo_label_df(input_label_path, mdet=True, attributes=defect_list)
    #         with_defect = (df[defect_list] > 0).any().any()
    #         if with_defect:
    #             image_name = stem2img_dict[Path(label_name).stem]
    #             defect_file_list.append(image_name)
    #
    #
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0912'
    # image_dir = os.path.join(root_dir, 'images')
    # label_dir = os.path.join(root_dir, 'labels')
    # defect_list = ['deformation', 'broken', 'abandonment', 'corrosion']
    # find_defect(label_dir, image_dir, defect_list)