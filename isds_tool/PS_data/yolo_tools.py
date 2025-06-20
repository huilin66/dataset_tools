import os
import shutil

from tqdm import tqdm
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from scipy import stats
import pandas as pd
import yaml

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
    for image_name in tqdm(image_list):
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

def random_select(data_dir, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True):
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
    df_train.to_csv(os.path.join(save_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, 'val.txt'), header=None, index=None)
    df_all.to_csv(os.path.join(save_dir, 'all.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, 'train.txt'),
                                           len(val_list), os.path.join(save_dir, 'val.txt')))

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

def data_merge(input_dir1, input_dir2, output_dir):
    print(f'merging {input_dir1} + {input_dir2} --> {output_dir}...')
    data_copy(input_dir1, output_dir)
    data_copy(input_dir2, output_dir)
    input_train_path1 = os.path.join(input_dir1, 'train.txt')
    input_train_path2 = os.path.join(input_dir2, 'train.txt')
    output_train_path = os.path.join(output_dir, 'train.txt')
    input_val_path1 = os.path.join(input_dir1, 'val.txt')
    input_val_path2 = os.path.join(input_dir2, 'val.txt')
    output_val_path = os.path.join(output_dir, 'val.txt')
    df_input_train1 = pd.read_csv(input_train_path1, names=['file_name'],header=None, index_col=False)
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
    print(f'merging {input_dir1} + {input_dir2} --> {output_dir} finished!')

def data_copy(input_dir, output_dir):
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

def get_yolo_label_df(gt_path, mdet=False, attributes=None):
    if mdet:
        assert attributes is not None, 'attribute_path must be provided, which is "%s"' % attributes
        if isinstance(attributes, str):
            attribute_keys = get_attributes(attributes)
        elif isinstance(attributes, list):
            attribute_keys = attributes
        names = ['category'] + ['attribute_len'] + attribute_keys + [ 'center_x', 'center_y', 'width', 'height']
    else:
        names = ['category', 'center_x', 'center_y', 'width', 'height']

    df = pd.DataFrame(None, columns=names + ['image'])
    with open(gt_path, 'r') as f:
        data = f.readlines()
        for id_line, line in enumerate(data):
            parts = line.strip().split(' ')
            category = int(parts[0])
            image_name = Path(gt_path).stem
            if mdet:
                att_len = int(parts[1])
                atts = list(map(float, parts[2:2 + att_len]))
                polygons = list(map(float, parts[2 + att_len:]))
                xywh = poly2xywh(polygons)
                df.loc[len(df)] = [category, att_len] + atts + xywh + [image_name]
            else:
                polygons = list(map(float, parts[1:]))
                xywh = poly2xywh(polygons)
                df.loc[len(df)] = [category] + xywh + [image_name]
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

if __name__ == '__main__':
    pass
    src_dir = r'/localnvme/data/billboard/ps_data/psdata118'
    mseg_dir = src_dir + '_mseg'
    mseg_c6_dir = src_dir + '_mseg_c6'
    seg_dir = src_dir + '_seg'
    seg_c6_dir = src_dir + '_seg_c6'

    # if os.path.exists(src_dir):
    #     os.rename(src_dir, mseg_dir)
    # mseg_class_update(mseg_dir, mseg_c6_dir)
    # mseg2seg(mseg_c6_dir, seg_c6_dir)
    #
    # random_select(mseg_c6_dir)
    # random_select(seg_c6_dir)


    # data_merge(r'/localnvme/data/billboard/ps_data/psdata118_mseg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata617_mseg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6')
    #
    # data_merge(r'/localnvme/data/billboard/ps_data/psdata118_seg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata617_seg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_seg_c6')
    #
    # data_merge(r'/localnvme/data/billboard/bd_data/data626_mseg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6',
    #            r'/localnvme/data/billboard/fused_data/data1361_mseg_c6')
    #
    # data_merge(r'/localnvme/data/billboard/bd_data/data626_seg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_seg_c6',
    #            r'/localnvme/data/billboard/fused_data/data1361_seg_c6')


    # mseg_attribute_update_gt(r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels_src',
    #                          r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels')
    # data_check(r'/localnvme/data/billboard/bd_data/data626_mseg_c6/labels',
    #            r'/localnvme/data/billboard/bd_data/data626_mseg_c6/attribute.yaml')

    # data_check(r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/attribute.yaml')
    # mseg_attribute_update_gt(r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels_src',
    #                          r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels')
    # mseg2seg(r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6', r'/localnvme/data/billboard/ps_data/psdata735_seg_c6')
    # mseg2seg(r'/localnvme/data/billboard/bd_data/data626_mseg_c6', r'/localnvme/data/billboard/bd_data/data626_seg_c6')

    # data_check(r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/attribute.yaml')
    # data_check(r'/localnvme/data/billboard/ps_data/psdata118_mseg_c6/labels',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/attribute.yaml')
    # data_check(r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/attribute.yaml')
    # data_check(r'/localnvme/data/billboard/fused_data/data1361_mseg_c6/labels',
    #            r'/localnvme/data/billboard/fused_data/data1361_mseg_c6/attribute.yaml')


    # data_merge(r'/localnvme/data/billboard/bd_data/data626_mseg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6',
    #            r'/localnvme/data/billboard/fused_data/data1361_mseg_c6')
    #
    # data_merge(r'/localnvme/data/billboard/bd_data/data626_seg_c6',
    #            r'/localnvme/data/billboard/ps_data/psdata735_seg_c6',
    #            r'/localnvme/data/billboard/fused_data/data1361_seg_c6')
