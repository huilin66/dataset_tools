import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from pathlib import Path

# ================= 核心工具函数 =================

def get_image_paths(folder_path):
    """获取文件夹中所有图像的路径，并按文件名排序"""
    return sorted([
        str(p) for p in Path(folder_path).iterdir()
        if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])

def preprocess_image(path, target_size=(256, 256)):
    """读取、调整大小并转灰度 (IO + 基础处理)"""
    try:
        # 使用 cv2.IMREAD_GRAYSCALE 直接读取为灰度，减少一次转换
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def calculate_ssim_pair(args):
    """
    计算一对图片的 SSIM (用于多进程)
    args: (img1_data, img2_data)
    """
    img1, img2 = args
    if img1 is None or img2 is None:
        return 0.0
    return ssim(img1, img2, data_range=255)

# ================= 核心逻辑类 =================

class DeduplicationTool:
    def __init__(self, input_dir, output_dir, window_size=10, threshold=0.5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.window_size = window_size
        self.threshold = threshold
        self.image_paths = get_image_paths(input_dir)
        self.n = len(self.image_paths)
        
    def _load_images_batch(self, indices, num_workers=1):
        """批量加载图片数据"""
        paths = [self.image_paths[i] for i in indices]
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # IO 密集型，使用线程池
                return list(executor.map(preprocess_image, paths))
        else:
            return [preprocess_image(p) for p in paths]

    def compute_similarity_matrix(self, num_workers=1):
        """
        计算稀疏相似度矩阵 (仅计算滑动窗口内的部分)
        返回: 完整的 N x N 矩阵 (未计算部分为 0)
        """
        matrix = np.zeros((self.n, self.n))
        print(f"Start computing SSIM matrix (Images: {self.n}, Window: {self.window_size})...")
        print(f"Mode: {'Multi-processing' if num_workers > 1 else 'Single-thread'}")

        # 预加载所有图片可能会爆内存，这里我们采用“分块处理”或者“滑动窗口动态加载”
        # 为了性能和实现的平衡，这里采用一次性预加载所有缩略图
        # (256x256 的灰度图仅 64KB，1000张才64MB，通常可以直接放入内存)
        print("Pre-loading images...")
        all_images = self._load_images_batch(range(self.n), num_workers=8 if num_workers > 1 else 1)
        
        tasks = []
        task_indices = []

        # 1. 准备任务列表
        for i in range(self.n):
            start = max(0, i - self.window_size)
            end = min(self.n, i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    # 只计算上三角，减少一半计算量
                    if i < j: 
                        tasks.append((all_images[i], all_images[j]))
                        task_indices.append((i, j))

        # 2. 执行计算
        results = []
        if num_workers > 1:
            # SSIM 是计算密集型，使用进程池
            # chunksize 对性能影响很大，任务很多时设大一点
            chunk_size = max(1, len(tasks) // (num_workers * 4))
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(calculate_ssim_pair, tasks, chunksize=chunk_size), 
                    total=len(tasks), 
                    desc="Computing SSIM"
                ))
        else:
            for task in tqdm(tasks, desc="Computing SSIM"):
                results.append(calculate_ssim_pair(task))

        # 3. 填回矩阵
        for (i, j), score in zip(task_indices, results):
            matrix[i][j] = score
            matrix[j][i] = score  # 对称矩阵

        return matrix

    def filter_and_copy(self, matrix):
        """筛选并拷贝"""
        print("Filtering images...")
        os.makedirs(self.output_dir, exist_ok=True)

        # 对角线置0
        np.fill_diagonal(matrix, 0)
        
        # 计算每张图在窗口范围内的最大相似度
        # 注意：这里 matrix 是稀疏计算的，未计算区域是0，不影响 max 逻辑
        max_similarities = np.max(matrix, axis=1)
        
        # 筛选逻辑：如果是去重，通常保留相似组里的一张，或者剔除相似度过高的
        # 根据你原本的逻辑：找出 "Unique" 的图片 (即与周围图片相似度都小于阈值)
        unique_indices = np.where(max_similarities <= self.threshold)[0]
        
        print(f"Found {len(unique_indices)} unique images (Threshold: {self.threshold})")
        
        for idx in tqdm(unique_indices, desc="Copying files"):
            src = self.image_paths[idx]
            dst = os.path.join(self.output_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    def plot_analysis(self, matrix):
        """可视化分析"""
        # 热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=False, cmap="YlOrRd")
        plt.title(f"Similarity Heatmap (Window: {self.window_size})")
        plt.show()

        # 曲线图
        np.fill_diagonal(matrix, 0)
        max_sims = np.max(matrix, axis=1)
        plt.figure(figsize=(12, 4))
        plt.plot(max_sims, label='Max Neighbor Similarity')
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        plt.legend()
        plt.title("Similarity Analysis")
        plt.show()

# ================= 外部调用接口 =================

def filter_deduplication_ssim(input_dir, output_dir, window_size=10, threshold=0.5, num_workers=1):
    """
    Args:
        num_workers: 
            1 = 单线程模式
            >1 = 多线程/多进程模式 (建议设为 CPU 核心数，例如 4 或 8)
    """
    tool = DeduplicationTool(input_dir, output_dir, window_size, threshold)
    
    # 1. 计算
    matrix = tool.compute_similarity_matrix(num_workers=num_workers)
    
    # 2. (可选) 可视化
    # tool.plot_analysis(matrix)
    
    # 3. 筛选和拷贝
    tool.filter_and_copy(matrix)

if __name__ == "__main__":
    # 使用示例
    # 单线程运行
    # filter_deduplication("data/input", "data/output", num_workers=1)
    
    # 多线程运行 (推荐)
    # filter_deduplication("data/input", "data/output", num_workers=8)
    pass