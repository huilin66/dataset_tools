import os
import cv2
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
def get_image_paths(folder_path):
    """获取文件夹中所有图像的路径，并按文件名排序"""
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

def compute_similarity_matrix(image_paths, window_size=5):
    """动态计算相似度矩阵，使用滑动窗口缓存避免重复读取"""
    n = len(image_paths)
    matrix = np.zeros((n, n))
    cache = {}  # 缓存图像数据：{index: preprocessed_image}

    for i in tqdm(range(n)):
        # 确定当前窗口范围 [start, end]
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        # 加载当前窗口内未缓存的图像
        for j in range(start, end):
            if j not in cache:
                img = cv2.imread(image_paths[j])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cache[j] = preprocess_image(img)  # 预处理并缓存

        # 计算当前图像i与窗口内其他图像的相似度
        for j in range(start, end):
            if i != j:
                matrix[i][j] = calculate_ssim(cache[i], cache[j])

        # 移出滑动窗口外的图像（释放内存）
        for j in list(cache.keys()):
            if j < i - window_size:
                del cache[j]

    return matrix

def preprocess_image(img, target_size=(256, 256)):
    """统一尺寸并转为灰度图"""
    img = cv2.resize(img, target_size)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def calculate_ssim(img1, img2):
    """计算两幅图像的SSIM相似度"""
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def plot_heatmap(matrix):
    """绘制热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="YlOrRd")
    plt.title("Similarity Heatmap (Sliding Window Cache)")
    plt.xlabel("Image Index")
    plt.ylabel("Image Index")
    plt.show()


import matplotlib.pyplot as plt


def plot_max_similarity_curve(similarity_matrix, threshold=0.7):
    """
    绘制每张图像的最大相似度曲线，并标注低于阈值的点
    Args:
        similarity_matrix: 相似度矩阵（n×n）
        threshold: 标记阈值（默认0.7）
    """
    # 计算每张图像的最大相似度（排除自相似）
    np.fill_diagonal(similarity_matrix, 0)
    max_similarities = np.max(similarity_matrix, axis=1)

    # 创建图像索引（x轴）
    image_indices = np.arange(len(max_similarities))

    # 绘制曲线和阈值线
    plt.figure(figsize=(12, 4))
    plt.plot(image_indices, max_similarities, 'b-', label='Max Similarity')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')

    # 标记低于阈值的点
    low_sim_indices = np.where(max_similarities <= threshold)[0]
    hig_sim_indices = np.where(max_similarities > threshold)[0]
    plt.scatter(low_sim_indices, max_similarities[low_sim_indices],
                c='red', s=50, zorder=5, label=f'Unique Images (≤ {threshold})')
    plt.scatter(hig_sim_indices, max_similarities[hig_sim_indices],
                c='green', s=50, zorder=5, label=f'Unique Images (> {threshold})')
    # 标注图像数量信息
    plt.title(f"Max Similarity Curve\n(Total unique images: {len(low_sim_indices)}/{len(max_similarities)})")
    plt.xlabel("Image Index")
    plt.ylabel("Max Similarity Score")
    plt.legend()
    plt.grid(True)
    plt.show()
def batch_filter_unique_images(image_paths, similarity_matrix, threshold=0.7, output_folder="unique_images"):
    """
    基于相似度矩阵批量筛选唯一图像（与任何其他图像的相似度均≤阈值）
    Args:
        image_paths: 图像路径列表
        similarity_matrix: 对称的相似度矩阵（n×n）
        threshold: 相似度阈值（默认0.7）
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1. 排除自相似（对角线置0）
    np.fill_diagonal(similarity_matrix, 0)

    # 2. 批量计算每张图像的最大相似度（与任何其他图像）
    max_similarities = np.max(similarity_matrix, axis=1)

    # 3. 筛选所有最大相似度≤阈值的图像索引
    unique_indices = np.where(max_similarities <= threshold)[0]

    # 4. 批量复制唯一图像
    for idx in unique_indices:
        src_path = image_paths[idx]
        dst_path = os.path.join(output_folder, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {os.path.basename(src_path)} (Max similarity: {max_similarities[idx]:.2f})")

    print(f"\nTotal unique images copied: {len(unique_indices)}")

def mannual_filter(input_dir, output_dir, filter_list):
    img_list = os.listdir(input_dir)
    img_list.sort()

    for idx, img_name in enumerate(tqdm(img_list)):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        if idx not in filter_list:
            shutil.copy(input_path, output_path)

def filter_deduplication(input_dir, output_dir, window_size=10, threshold=0.5):
    pass
    image_paths = get_image_paths(input_dir)
    similarity_matrix = compute_similarity_matrix(image_paths, window_size=window_size)
    batch_filter_unique_images(image_paths, similarity_matrix, threshold=threshold, output_folder=output_dir)

if __name__ == '__main__':
    pass
    # 主流程
    # folder_path = r"E:\data\202502_signboard\PS\20250616\selected_img"
    # output1_path = r"E:\data\202502_signboard\PS\20250616\selected_img_filter1"
    # output2_path = r"E:\data\202502_signboard\PS\20250616\selected_img_filter2"
    # # image_paths = get_image_paths(folder_path)
    # # similarity_matrix = compute_similarity_matrix(image_paths, window_size=10)
    # # # plot_heatmap(similarity_matrix)
    # # plot_max_similarity_curve(similarity_matrix, threshold=0.5)
    # # batch_filter_unique_images(image_paths, similarity_matrix, threshold=0.5, output_folder=output1_path)
    #
    # mannual_filter(output1_path, output2_path, list(range(84,98))+list(range(119,278)))

    filter_deduplication(
        input_dir=r'Y:\ZHL\isds\PS\task0606\angle6\rectified_image_select',
        output_dir=r'Y:\ZHL\isds\PS\task0606\angle6\rectified_image_filter', window_size=10, threshold=0.5)
    # folder_path = r"E:\data\202502_signboard\PS\20250616\rectified_image2\selected_img"
    # output1_path = r"E:\data\202502_signboard\PS\20250616\rectified_image2\selected_img_filter1"
    # output2_path = r"E:\data\202502_signboard\PS\20250616\rectified_image2\selected_img_filter2"
    # image_paths = get_image_paths(folder_path)
    # similarity_matrix = compute_similarity_matrix(image_paths, window_size=10)
    # plot_max_similarity_curve(similarity_matrix, threshold=0.5)
    # batch_filter_unique_images(image_paths, similarity_matrix, threshold=0.5, output_folder=output1_path)

    # mannual_filter(output1_path, output2_path, list(range(84,98))+list(range(119,278)))