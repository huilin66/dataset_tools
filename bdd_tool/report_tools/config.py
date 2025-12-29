# config.py
import os

# --- 1. 路径配置 (根据你的环境修改) ---
# 数据集根目录
ROOT_DIR = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12'

# 具体子目录
IMG_DIR = os.path.join(ROOT_DIR, 'val', 'images')
PRED_DIR = os.path.join(ROOT_DIR, 'result_analysis', 'val_infer', 'labels')
CLASS_PATH = os.path.join(ROOT_DIR, 'classes.txt')

# 输出文件路径
OUTPUT_PDF_PATH = os.path.join(ROOT_DIR, 'result_analysis', 'val_infer', 'report_modular.pdf')

# --- 2. 字体配置 (解决中文或样式问题) ---
# Windows 默认路径，Linux下需要修改
FONT_PATH_REGULAR = r"C:\Windows\Fonts\times.ttf"
FONT_PATH_BOLD = r"C:\Windows\Fonts\timesbd.ttf"

# --- 3. 业务逻辑阈值 ---
# 缺陷等级判定阈值 [pixel]
# 宽度或高度超过 50 算 Moderate，超过 500 算 Serious
LEVELS_THRESHOLD = [50, 500] 


# --- 4. 可视化配置 ---
# 默认颜色列表 (BGR 格式或 RGB 格式均可，取决于你的绘图函数逻辑)
# 这里定义为 RGB
COLOR_PALETTE = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
] * 10  # 复制多次以防类别过多

# --- 5. 地理/楼层计算配置 ---
# 格式: [地面基础高度, 裙楼1高, 裙楼2高..., 标准层高, 顶层高]
# 默认: 地面0m, 只有1层裙楼(10m), 标准层3m, 顶层5m
FLOOR_CONFIG = [0, 10, 3, 5]

# --- 6. 相机硬件参数 (用于 GSD 计算) ---
# DJI M4T / M30T Wide Camera 通常是 1/2 英寸 CMOS
# 1/2 英寸传感器宽度约为 6.4mm (具体值: 6.40mm x 4.80mm)
# 4/3 英寸传感器宽度约为 17.3mm
SENSOR_WIDTH_MM = 6.4 

# 如果有不同机型，可以建立字典映射
# DRONE_SENSOR_MAP = {
#     'M4T': 6.4,
#     'M3E': 17.3
# }

# --- 7. 缺失数据的兜底参数 ---
# 默认拍摄距离 (当 XMP 丢失时使用)
DEFAULT_DISTANCE_M = 15.0

# 默认焦距 (当 EXIF 丢失时使用) [单位: mm]
# 热成像(_T)通常读不到EXIF，建议设置为热成像镜头的物理焦距 (例如 13.5mm 或 9.1mm，取决于机型)
# 如果是 M4T/M3T 的广角镜头，通常是 4.5mm
DEFAULT_FOCAL_LENGTH_MM = 13.5