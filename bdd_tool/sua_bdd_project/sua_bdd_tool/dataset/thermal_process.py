import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ================= 配置区域 =================
# 输入文件夹
SOURCE_FOLDER = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\images_crop\high"
DST_FOLDER = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\check\rule_check"
# 输出的四个分类文件夹名称
DIR_NAMES = {
    "yellow": "1_Has_Yellow",       # 有一些黄色
    "orange": "2_Half_Orange",      # 没黄，一半橙
    "red":    "3_Mostly_Red",       # 基本都是红
    "shadow": "4_Shadow",          # 比较暗
    "dark":   "5_Dark"          # 基本都是暗红/其他
}

# 判定阈值 (0.01 代表 1%)
TH_YELLOW = 0.10  #  5%：只要有一点点黄色就算
TH_ORANGE = 0.33   # 45%：接近一半是橙色
TH_RED    = 0.66   # 50%：红色占主导地位
TH_SHADOW = 0.33   # 33%：红色占主导地位
# ===========================================

def get_color_ratios(img):
    """
    计算黄、橙、红在图片中的占比
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = img.shape[0] * img.shape[1]

        # 1. 定义颜色范围 (OpenCV HSV: H=0-179)
        # 黄色 H: 20-35
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        
        # 橙色 H: 11-19
        lower_orange = np.array([11, 50, 50])
        upper_orange = np.array([19, 255, 255])
        
        # 红色 H: 0-10 和 170-179
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([179, 255, 255])

        # 2. 计算各颜色掩膜像素数
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        count_yellow = np.count_nonzero(mask_yellow)

        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        count_orange = np.count_nonzero(mask_orange)

        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        count_red = np.count_nonzero(mask_red)

        # 3. 返回占比
        return (count_yellow / total_pixels, 
                count_orange / total_pixels, 
                count_red / total_pixels)

    except Exception as e:
        print(f"颜色分析出错: {e}")
        return 0, 0, 0

def classify_and_copy(src_dir, dst_dir):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 创建4个目标文件夹
    target_dirs = {}
    for key, name in DIR_NAMES.items():
        p = dst_path / name
        p.mkdir(parents=True, exist_ok=True)
        target_dirs[key] = p

    # 获取所有图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(src_path.glob(ext))
    
    print(f"正在处理 {len(files)} 张图片...\n")

    stats = {k: 0 for k in DIR_NAMES.keys()}

    for file_path in files:
        # 读取图片
        # 遇到中文路径建议用 cv2.imdecode，这里用标准读取
        img = cv2.imread(str(file_path))
        if img is None:
            continue

        # 获取颜色占比
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # ============ 核心分类逻辑 (优先级从高到低) ============
        target_key = ""
        reason = ""

        # 1. 优先检查黄色
        if r_yellow > TH_YELLOW:
            target_key = "yellow"
            reason = f"黄色占比 {r_yellow:.2%} > {TH_YELLOW}"
        
        # 2. 其次检查橙色 (前提是没进黄色组)
        elif r_orange > TH_ORANGE:
            target_key = "orange"
            reason = f"橙色占比 {r_orange:.2%} > {TH_ORANGE}"
            
        # 3. 再次检查红色
        elif r_red > TH_RED:
            target_key = "red"
            reason = f"红色占比 {r_red:.2%} > {TH_RED}"
        
        elif r_red > TH_SHADOW:
            target_key = "shadow"
            reason = f"红色占比 {r_red:.2%} > {TH_SHADOW}"

        # 4. 剩下的归为暗红/其他
        else:
            target_key = "dark"
            reason = "无显著黄/橙/红特征"

        # ====================================================

        # 执行移动
        dest_folder = target_dirs[target_key]
        try:
            shutil.copy(str(file_path), str(dest_folder / file_path.name))
            stats[target_key] += 1
            print(f"[移动到 {DIR_NAMES[target_key]}] {file_path.name} ({reason})")
        except Exception as e:
            print(f"移动文件 {file_path.name} 失败: {e}")

    # 打印总结
    print("\n" + "="*30)
    print("分类完成统计：")
    for key, count in stats.items():
        print(f"{DIR_NAMES[key]}: {count} 张")
    print("="*30)


def extract_raw_thermal_single(img_path: Path, output_dir: Path):
    """
    单个文件处理函数：提取 RawThermalImage 并保存为 TIFF
    """
    try:
        # 构建输出文件名：保持原文件名，后缀改为 .tiff
        # 例如: DJI_0001.JPG -> output_dir/DJI_0001.tiff
        output_path = output_dir / img_path.with_suffix('.tiff').name
        
        # 如果目标文件已存在，跳过（可选）
        if output_path.exists():
            return True

        # 调用 ExifTool 提取二进制数据
        # -b: 二进制模式
        # -RawThermalImage: 目标标签
        cmd = ["exiftool", "-b", "-RawThermalImage", str(img_path)]
        
        # 运行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, check=False)
        
        # 如果提取到了数据（result.stdout 不为空）
        if result.stdout:
            with open(output_path, "wb") as f:
                f.write(result.stdout)
            return True
        else:
            # 说明这张图里没有 RawThermalImage 标签（可能不是红外图，或者是可见光图）
            return False
            
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return False

def batch_extract_thermal(input_root: str, output_root: str, max_workers: int = 8):
    """
    批量提取程序
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 收集所有 JPG 文件
    print(f"正在扫描 {input_root} 下的图片...")
    # 使用 rglob 递归查找所有 .JPG 或 .jpg
    all_images = list(input_path.rglob("*.[jJ][pP][gG]"))
    
    print(f"找到 {len(all_images)} 张图片，准备开始提取 Raw 数据...")

    # 2. 多线程处理
    success_count = 0
    fail_count = 0 # 指没有Raw数据的图片（比如可见光照片）

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = []
        for img in all_images:
            # 这里的逻辑是：所有JPG都试一遍，ExifTool 只有在找到 Tag 时才会输出文件
            future = executor.submit(extract_raw_thermal_single, img, output_path)
            futures.append(future)

        # 使用 tqdm 显示进度
        for future in tqdm(futures, desc="Extracting Raw Thermal"):
            if future.result():
                success_count += 1
            else:
                fail_count += 1

    print("\n" + "="*30)
    print("提取完成！")
    print(f"成功提取 Raw TIFF: {success_count} 张")
    print(f"跳过 (无热成像数据): {fail_count} 张")
    print(f"输出目录: {output_path}")
    print("="*30)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_iron_cmap():
    """
    手动创建 Iron/IronBow 颜色映射
    """
    # 定义锚点：(位置, (R, G, B))
    # 注意：Matplotlib 需要颜色归一化到 0-1
    colors = [
        (0.0,  (0/255, 0/255, 0/255)),     # Black
        (0.2,  (0/255, 0/255, 140/255)),   # Blue
        (0.4,  (145/255, 0/255, 145/255)), # Magenta
        (0.6,  (255/255, 0/255, 0/255)),   # Red
        (0.8,  (255/255, 180/255, 0/255)), # Orange/Yellow
        (1.0,  (255/255, 255/255, 255/255)) # White
    ]
    
    # 创建线性插值的 Colormap
    cmap_name = 'iron_red'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

def apply_iron_colormap(raw_data, vmin=None, vmax=None):
    """
    将原始 2D 数据转换为 IronRed RGB 图像
    """
    if vmin is None: vmin = np.min(raw_data)
    if vmax is None: vmax = np.max(raw_data)
    
    # 1. 归一化数据到 0-1 之间
    # 避免除以零
    denom = (vmax - vmin) + 1e-6
    normalized = (raw_data - vmin) / denom
    normalized = np.clip(normalized, 0, 1)
    
    # 2. 获取 Colormap
    iron_cmap = create_iron_cmap()
    
    # 3. 应用映射 (Matplotlib 会返回 RGBA, 我们取前3个通道 RGB)
    colored_image = iron_cmap(normalized)[:, :, :3]
    
    # 转换回 0-255 的 uint8 格式以保存图片
    colored_image_uint8 = (colored_image * 255).astype(np.uint8)
    
    return colored_image_uint8

# # --- 测试示例 ---
# if __name__ == "__main__":
#     # 模拟一个从 10度 到 50度 的温度矩阵 (梯度图)
#     dummy_temp_data = np.linspace(10, 50, 256*256).reshape(256, 256)
    
#     # 生成 IronRed 图像
#     rgb_img = apply_iron_colormap(dummy_temp_data, vmin=10, vmax=50)
    
#     # 显示
#     plt.figure(figsize=(6, 6))
#     plt.title("IronRed Mapping Simulation")
#     plt.imshow(rgb_img)
#     plt.axis('off')
#     plt.show()

#     # 如果你需要导出 LUT 数组给 C++ 或 OpenCV 使用：
#     iron_cmap = create_iron_cmap()
#     lut = (iron_cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
#     print("LUT Shape:", lut.shape) # (256, 3) -> 每一行是一个 (R, G, B)
if __name__ == "__main__":
    classify_and_copy(SOURCE_FOLDER, DST_FOLDER)