import os
import os.path as osp
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def seg_filter_small(input_dir, output_dir, threshold=0.01, class_list=[2,4,6,7], with_attribute=True):
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
        input_image_path = os.path.join(input_image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        shutil.copy(input_image_path, output_image_path)

        area_poly, area_bbox = filter_yolo_segmentation(input_label_path, output_label_path, threshold=threshold, with_attribute=with_attribute, class_list=class_list)
        areas_poly += area_poly
        areas_bbox += area_bbox
    analyze_area_distribution(areas_poly, os.path.join(output_dir, 'areas_poly.png'))
    analyze_area_distribution(areas_bbox, os.path.join(output_dir, 'areas_bbox.png'))

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

if __name__ == '__main__':
    pass

    # seg_filter_small(r'E:\data\202502_signboard\data_annotation\annotation_result_merge',
    #                  r'E:\data\202502_signboard\data_annotation\annotation_result_merge_f001', with_attribute=True, threshold=0.01)
    # seg_filter_small(r'E:\data\202502_signboard\data_annotation\annotation_result_merge',
    #                  r'E:\data\202502_signboard\data_annotation\annotation_result_merge_f010', with_attribute=True, threshold=0.1)


    seg_filter_small(r'/localnvme/data/billboard/fused_data/data870_mseg_c6',
                     r'/localnvme/data/billboard/fused_data/data870_mseg_c6_f010', with_attribute=True, threshold=0.10)
    seg_filter_small(r'/localnvme/data/billboard/fused_data/data870_seg_c6',
                     r'/localnvme/data/billboard/fused_data/data870_seg_c6_f010', with_attribute=False, threshold=0.10)