import os
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
import platform
import json
import math
import subprocess
import time

# ===================== 颜色工具 =====================
# 定义一个高对比度的调色盘 (R, G, B)
COLOR_PALETTE = [
    (255, 50, 50),    # 0: Red
    (50, 255, 50),    # 1: Green
    (50, 50, 255),    # 2: Blue
    (255, 255, 50),   # 3: Yellow
    (50, 255, 255),   # 4: Cyan
    (255, 50, 255),   # 5: Magenta
    (255, 128, 0),    # 6: Orange
    (128, 0, 255),    # 7: Purple
    (0, 128, 128),    # 8: Teal
    (128, 128, 0)     # 9: Olive
]

def get_class_color(cls_idx):
    """根据类别索引返回颜色"""
    if cls_idx < 0: return (255, 255, 255)
    return COLOR_PALETTE[cls_idx % len(COLOR_PALETTE)]

def get_contrasting_text_color(bg_color):
    """根据背景色亮度决定文字是黑还是白"""
    luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)


# ===================== 0. 辅助函数：加载类别名称 =====================
def load_class_names(class_path):
    """加载 classes.txt，按行号对应类别ID"""
    if not class_path or not os.path.exists(class_path):
        print("⚠️ 未找到 classes.txt，将仅显示类别 ID")
        return None
    with open(class_path, "r", encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    return names

# ===================== 1. 工具函数 =====================
def parse_yolo_line(line):
    vals = list(map(float, line.strip().split()))
    # 增强健壮性：只要大于等于5列都尝试解析
    if len(vals) >= 5:
        cls, cx, cy, w, h = vals[:5]
        conf = vals[5] if len(vals) >= 6 else 1.0
        return int(cls), cx, cy, w, h, conf
    else:
        return None

def yolo_to_pixel(cx, cy, w, h, W, H):
    return cx * W, cy * H, w * W, h * H

def project_to_facade(px, py, W, H, proj_params):
    fy = H * (proj_params["focal_length_mm"] / proj_params["sensor_height_mm"])
    z = proj_params["camera_height_m"] + (H / 2 - py) / fy * proj_params["facade_distance_m"]
    x = (px - W / 2) / fy * proj_params["facade_distance_m"]
    return x, z

def iou_1d(a_min, a_max, b_min, b_max):
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    return inter / union if union > 0 else 0

def compute_iou_2d(box1, box2):
    """
    计算两个像素矩形的 IoU
    box: (cx, cy, w, h)
    """
    x1_min = box1[0] - box1[2]/2
    x1_max = box1[0] + box1[2]/2
    y1_min = box1[1] - box1[3]/2
    y1_max = box1[1] + box1[3]/2

    x2_min = box2[0] - box2[2]/2
    x2_max = box2[0] + box2[2]/2
    y2_min = box2[1] - box2[3]/2
    y2_max = box2[1] + box2[3]/2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def pyexif_to_dict(img_path):
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()

def parse_dji_float(val_str):
    """
    解析 DJI 格式的数字字符串，例如 '+60.816', '7.233 m', '24.0 mm'
    """
    if isinstance(val_str, (int, float)):
        return float(val_str)
    try:
        # 去掉非数字字符（除了 . + -）
        clean_str = "".join([c for c in str(val_str) if c in "0123456789.+-"])
        return float(clean_str)
    except:
        return None

def get_image_metadata(img_path):
    """
    调用 exiftool 获取元数据 (模拟你提到的 pyexif_to_dict)
    如果你有自己的函数，请替换这里
    """
    # 这里为了演示，使用 subprocess 调用 exiftool 命令行
    # 实际使用中，请替换为你项目中现有的 pyexif_to_dict 调用
    try:
        cmd = ['exiftool', '-j', '-G', img_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout)[0]
        
        # 提取关键参数
        # 注意：exiftool 的键名可能带前缀，如 'XMP:RelativeAltitude' 或直接 'RelativeAltitude'
        # 这里做一个简单的查找映射
        def find_val(keys):
            for k in keys:
                for full_k in info.keys():
                    if k in full_k: # 模糊匹配
                        return info[full_k]
            return None

        # 1. 高度 (Z轴核心)
        # [修改点 1] 优先读取 AbsoluteAltitude
        # 注意：DJI M4T 的 AbsoluteAltitude 通常非常准 (RTK)
        abs_alt = parse_dji_float(find_val(['AbsoluteAltitude', 'GPSAltitude']))
        rel_alt = parse_dji_float(find_val(['RelativeAltitude']))
        
        # 逻辑：如果能读到绝对海拔，就用绝对的；否则降级用相对的（作为兜底）
        # 但记得在外面处理时，如果用的是相对高度，street_level 就应该设为 0
        final_alt = abs_alt if abs_alt is not None else rel_alt
        
        # 2. 激光测距 (距离核心)
        lrf = parse_dji_float(find_val(['LRFTargetDistance']))
        
        # 3. 云台俯仰 (角度补偿)
        pitch = parse_dji_float(find_val(['GimbalPitchDegree', 'FlightPitchDegree']))
        
        # 4. 焦距 (自动获取，不再硬编码)
        # 优先使用等效35mm焦距，或者使用实际焦距配合传感器尺寸计算
        focal_35 = parse_dji_float(find_val(['FocalLengthIn35mmFormat', 'FocalLength35efl']))
        
        return {
            "alt": final_alt if final_alt is not None else 0.0,
            "is_absolute": (abs_alt is not None),
            "lrf": lrf if lrf is not None else 10.0, # 默认值防止崩溃
            "pitch": pitch if pitch is not None else 0.0,
            "focal_35mm": focal_35 if focal_35 is not None else 24.0
        }
    except Exception as e:
        print(f"⚠️ 读取 EXIF 失败 {img_path}: {e}")
        return {"alt": 0, "lrf": 10, "pitch": 0, "focal_35mm": 24}

class FloorManager:
    def __init__(self, floor_params):
        self.params = floor_params
        self.floor_map = {} # {'1/F': (start_z, end_z), ...}
        self.is_valid = False
        self._parse_and_build()
        self.print_floor_chart()

    def _parse_and_build(self):
        p = self.params
        
        # 1. 单位换算 (检测 normal floor height 是否大于 100)
        # 如果大于 100，说明是毫米，scale = 0.001；否则是米，scale = 1.0
        scale = 0.001 if p['normal floor height'] > 100 else 1.0
        
        base_h = p['base_height'] * scale
        final_h = p['final height'] * scale
        norm_h = p['normal floor height'] * scale
        
        # 转换列表和字典中的高度
        podium_hs = [h * scale for h in p['podium heights']]
        top_hs = [h * scale for h in p['top heights']]
        special_hs = {str(k): v * scale for k, v in p['special heights'].items()}
        
        # 2. 构建楼层序列 (Name, Height)
        floor_sequence = []
        
        # A. Podium (裙楼/底层)
        if len(p['podium names']) != len(podium_hs):
            print(f"❌ 楼层参数错误: Podium 名字数量 ({len(p['podium names'])}) 与 高度数量 ({len(podium_hs)}) 不一致")
            return
            
        for name, h in zip(p['podium names'], podium_hs):
            floor_sequence.append((str(name), h))
            
        # B. Normal (标准层 + 特殊层)
        # range 是左闭右闭，所以 end + 1
        start_idx, end_idx = p['normal height number list']
        expected_norm_count = p['normal height numbers']
        
        # 校验数量
        real_norm_count = end_idx - start_idx + 1
        if real_norm_count != expected_norm_count:
             print(f"⚠️ 警告: Normal floor 数量定义不一致 (Number: {expected_norm_count} vs List range: {real_norm_count})，以 List 为准")

        for i in range(start_idx, end_idx + 1):
            name = str(i)
            # 检查是否是特殊层
            h = special_hs.get(name, norm_h)
            floor_sequence.append((name, h))
            
        # C. Top (顶层)
        if len(p['top names']) != len(top_hs):
            print(f"❌ 楼层参数错误: Top 名字数量 ({len(p['top names'])}) 与 高度数量 ({len(top_hs)}) 不一致")
            return

        for name, h in zip(p['top names'], top_hs):
            floor_sequence.append((str(name), h))

        # 3. 生成高度分布字典 & 校验总高度
        current_z = base_h
        
        for name, h in floor_sequence:
            # 格式化 Key: "楼层编号/F"
            key = f"{name}/F"
            self.floor_map[key] = (current_z, current_z + h)
            current_z += h
            
        # 4. 校验高度闭环
        # 理论总高度 = final - base
        # 累加总高度 = current_z - base_h
        self.final_calc_height = current_z
        diff = abs(current_z - final_h)
        
        print(f"🏢 楼层构建完成: 起始 {base_h:.2f}m -> 计算结束 {current_z:.2f}m (定义结束 {final_h:.2f}m)")
        
        if diff > 0.1: # 允许 10cm 误差
            print(f"⚠️ 警告: 建筑高度校验失败! 偏差 {diff:.4f}m")
            print("   请检查: base_height, final height 或 各层高度之和是否匹配")
        else:
            print("✅ 建筑高度校验通过")
            self.is_valid = True

    def get_floor(self, z_value):
        """根据绝对海拔 Z 返回楼层名"""
        # 允许一定的容差，处理刚好踩线的情况
        epsilon = 0.01 
        
        for name, (start, end) in self.floor_map.items():
            if start - epsilon <= z_value < end + epsilon:
                return name
        
        # 如果找不到
        sorted_floors = sorted(self.floor_map.values(), key=lambda x: x[0])
        if not sorted_floors: return "Unknown"
        
        min_h = sorted_floors[0][0]
        max_h = sorted_floors[-1][1]
        
        if z_value < min_h:
            return "Below Base"
        elif z_value >= max_h:
            return "Above Top"
        
        return "Unknown"
    # ==================== [新增] 可视化打印方法 ====================
    def print_floor_chart(self):
        """在控制台打印楼层高度标尺"""
        if not self.floor_map:
            return

        print("\n🏢 Building Elevation Chart (Top-Down)")
        print("=" * 40)
        
        # 1. 打印最顶部的线 (Final Height)
        print(f"{'[TOP]':<10} ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅  {self.final_calc_height:7.2f}m")

        # 2. 获取所有楼层并按高度倒序排列
        # floor_map value is (start, end), we sort by start descending
        sorted_floors = sorted(self.floor_map.items(), key=lambda item: item[1][0], reverse=True)

        for name, (start_z, end_z) in sorted_floors:
            # 打印每一层的起始高度 (Floor Level)
            # 格式：名字占10格，下划线，高度
            print(f"{name:<10} ______  {start_z:7.2f}m")
            
        # 3. 打印基底 (Base Height)
        # 通常最底层的 Start Z 就是 Base，但为了明确，再打一行 Base
        # 如果最底层名字不是 BASE，这行很有用
        print(f"{'[BASE]':<10} ______  {self.params['base_height'] * (0.001 if self.params['base_height']>100 else 1.0):7.2f}m")
        print("=" * 40 + "\n")
    def get_floor_info(self):
        """返回生成的字典供 JSON 导出"""
        return self.floor_map
    def write_floor_map(self, output_path):
        """将楼层映射写入 JSON 文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.floor_map, f, indent=4, ensure_ascii=False)

def export_projection_details_json(all_dets, output_path):
    print(f"📊 正在导出投影详情 JSON: {output_path} ...")
    
    # 按图片分组
    data_by_img = {}
    for d in all_dets:
        img_name = d["img"]
        data_by_img.setdefault(img_name, [])
        
        # 构造单条记录
        entry = {
            "gid": d["gid"],           # 全局唯一流水号
            "cls": int(d["cls"]),      # 类别索引
            "id": d.get("id", -1),     # 最终分配的去重ID
            "conf": float(f"{d['conf']:.4f}"),
            "floor": d["floor"],

            # === 原始数据 ===
            "raw_yolo": {
                "cx": float(f"{d['cxcywh'][0]:.6f}"),
                "cy": float(f"{d['cxcywh'][1]:.6f}"),
                "w":  float(f"{d['cxcywh'][2]:.6f}"),
                "h":  float(f"{d['cxcywh'][3]:.6f}")
            },
            
            # === 像素数据 (方便在图中画框核对) ===
            "pixel": {
                "px": float(f"{d['pxpywh'][0]:.2f}"),
                "py": float(f"{d['pxpywh'][1]:.2f}"),
                "bw": float(f"{d['pxpywh'][2]:.2f}"),
                "bh": float(f"{d['pxpywh'][3]:.2f}")
            },
            
            # === 投影数据 (排查去重逻辑的关键) ===
            "projection_world": {
                "x (horizontal_m)": float(f"{d['x']:.4f}"), # 水平距离（用于区分左右物体）
                "z (height_m)":     float(f"{d['z']:.4f}"), # 垂直高度
                "h (obj_height_m)": float(f"{d['h']:.4f}")  # 物体实际高度
            }
        }
        data_by_img[img_name].append(entry)

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_by_img, f, indent=4, ensure_ascii=False)
    
    print("✅ 投影详情导出完成。")

def calculate_robust_wall_distance(img_files, trim_ratio=0.05, bin_size=0.5):
    """
    更加鲁棒的墙面距离计算函数
    :param img_files: 图片路径列表
    :param trim_ratio: 首尾截断比例 (0.1 表示去掉前10%和后10%)
    :param bin_size: 直方图分桶大小 (单位: 米)，建议 0.5m 或 1.0m
    """
    print("📏 正在分析采集路线 (高阶去噪版)...")
    
    # 1. 确保按拍摄顺序排列 (假设文件名包含时间或序号)
    sorted_files = sorted(img_files)
    total_imgs = len(sorted_files)
    
    if total_imgs == 0:
        return 10.0

    # 2. 读取所有 LRF 数据
    valid_lrf = []
    
    # 为了演示，这里假设已经有了 get_image_metadata 函数
    # 在实际循环中读取
    raw_data = []
    for img_path in tqdm(sorted_files, desc="Reading Metadata"):
        meta = get_image_metadata(img_path)
        dist = meta['lrf']
        # 过滤明显的错误数据 (比如 < 1m 或 > 100m)
        if 1.0 < dist < 100.0:
            raw_data.append(dist)
        else:
            raw_data.append(None) # 保持索引对应，方便截断

    # 3. 首尾截断 (Head/Tail Trimming)
    # 计算需要截掉的数量
    trim_cnt = int(total_imgs * trim_ratio)
    
    # 截取中间段
    if total_imgs > 2 * trim_cnt:
        trimmed_data = raw_data[trim_cnt : total_imgs - trim_cnt]
        print(f"✂️ 已剔除首尾各 {trim_cnt} 张图片，保留中间 {len(trimmed_data)} 张")
    else:
        trimmed_data = raw_data
        print("⚠️ 图片过少，跳过首尾截断")

    # 去除 None 值
    clean_lrf = [x for x in trimmed_data if x is not None]
    
    if not clean_lrf:
        print("❌ 有效 LRF 数据不足，使用默认值 10m")
        return 10.0

    # 4. 基于直方图寻找“众数区间” (Mode Binning)
    # 这是处理浮点数众数的最佳方法
    
    # 创建分桶区间：从最小值到最大值，步长为 bin_size
    min_val = min(clean_lrf)
    max_val = max(clean_lrf)
    bins = np.arange(math.floor(min_val), math.ceil(max_val) + bin_size, bin_size)
    
    # 统计直方图
    hist, bin_edges = np.histogram(clean_lrf, bins=bins)
    
    # 找到数量最多的那个桶 (Peak Index)
    peak_idx = np.argmax(hist)
    
    # 获取该桶的范围
    peak_start = bin_edges[peak_idx]
    peak_end = bin_edges[peak_idx+1]
    
    print(f"📊 发现主墙面区间: {peak_start:.2f}m ~ {peak_end:.2f}m (包含 {hist[peak_idx]} 张图片)")
    
    # 5. 在“主墙面区间”内计算精确中位数
    # 这一步是为了防止桶太大导致精度不够，或者桶太小导致切分错误
    # 我们只选取落在主区间内的数据来算最终结果
    final_candidates = [x for x in clean_lrf if peak_start <= x < peak_end]
    
    if not final_candidates:
        # 理论上不会发生，除非 histogram 逻辑出错，兜底用整体中位数
        final_dist = np.median(clean_lrf)
    else:
        final_dist = np.median(final_candidates)

    print(f"✅ 最终选定墙面基准距离: {final_dist:.4f}m (基于众数区间优化)")
    return final_dist

def calculate_global_wall_distance(img_files):
    """
    统计所有图片的 LRF 距离，计算中位数，作为“墙面基准距离”
    """
    print("📏 正在分析采集路线的墙面距离统计特征...")
    lrf_values = []
    
    # 随机抽样 20 张或者全部读取来计算，为了速度可以抽样
    # 这里为了准确，全部读取
    for img_path in tqdm(img_files):
        meta = get_image_metadata(img_path)
        if meta['lrf'] > 1.0: # 过滤掉无效值（如0或极小值）
            lrf_values.append(meta['lrf'])
            
    if not lrf_values:
        return 10.0 # 默认兜底
        
    median_dist = np.median(lrf_values)
    mean_dist = np.mean(lrf_values)
    std_dist = np.std(lrf_values)
    
    print(f"📊 距离统计: 中位数={median_dist:.2f}m, 均值={mean_dist:.2f}m, 标准差={std_dist:.2f}m")
    print(f"✅ 将使用中位数 {median_dist:.2f}m 作为固定墙面距离 (去除窗户/噪点影响)")
    return median_dist

def project_adaptive(px, py, W, H, meta, wall_distance_m):
    """
    自适应投影函数
    :param meta: 当前图片的元数据 (alt, pitch, focal_35mm)
    :param wall_distance_m: 全局固定的墙面距离
    """
    # 1. 计算像素焦距 (基于 35mm 等效焦距)
    # 35mm 全画幅传感器宽度为 36mm
    # fx_pix = (F_35mm / 36mm) * ImageWidth_pix
    fx = (meta['focal_35mm'] / 36.0) * W
    fy = fx # 假设像素是正方形
    
    # 2. 归一化像素坐标 (以图像中心为原点)
    u = px - W / 2
    v = H / 2 - py # 图像向上为y正方向
    
    # 3. 角度计算 (引入云台 Pitch 修正)
    # alpha_y 是像素点相对于光轴的垂直夹角
    alpha_y = math.atan(v / fy) 
    
    # 实际视线与水平面的夹角 = 云台Pitch + 像素夹角
    # 注意：DJI GimbalPitch 向上为正还是向下为正？通常水平是0，向下是负。
    # 根据你的数据 '+0.00'，假设向上为正。
    # 修正：通常俯视拍摄，Pitch是负的。如果 Pitch 是 +0.0，说明是水平拍摄。
    theta_total = math.radians(meta['pitch']) + alpha_y
    
    # 4. 计算物理坐标
    # Z (高度) = 无人机高度 + 垂直增量
    # 垂直增量 = 距离 * tan(总角度)
    z = meta['alt'] + wall_distance_m * math.tan(theta_total)
    
    # X (水平) = 距离 * tan(水平夹角) / cos(垂直夹角修正)
    # 简单近似：x = u / fx * distance
    x = (u / fx) * wall_distance_m
    
    # H (物体高度) = 像素高 / fy * 距离
    # 严格来说也受 pitch 影响，但物体很小时可忽略
    h_obj = (py / H) * 0 # 这是一个占位，实际应该用 box_h
    
    return x, z

# ===================== 2. 核心处理流程 =====================
def yolo_project2facade_adaptive(img_dir, yolo_txt_dir, target_classes=None, floor_params=None):
    all_dets = []
    gid = 0

    # 1. 初始化楼层管理器
    print("🏗️ 正在初始化楼层数据...")
    floor_mgr = FloorManager(floor_params)
    floor_mgr.write_floor_map("floor_map.json")
    
    if not floor_mgr.is_valid:
        print("⚠️ 楼层参数校验未通过，楼层计算可能不准确")


    # 1. 获取所有图片列表
    img_files = glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg"))
    
    if not img_files:
        print("❌ 未找到图片")
        return []

    # 2. 【关键步骤】预计算全局墙面距离
    global_wall_dist = calculate_robust_wall_distance(img_files)

    print("📥 读取 YOLO + 自适应投影 (Metadata Driven)...")

    for img_path in tqdm(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(yolo_txt_dir, name + ".txt")
        
        if not os.path.exists(txt_path):
            continue

        # 读取图片尺寸
        with Image.open(img_path) as img:
            W, H = img.size
            
        # 【关键步骤】读取当前图片的元数据
        meta = get_image_metadata(img_path)
        
        with open(txt_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) < 5: continue
                
                cls = int(vals[0])
                if target_classes and cls not in target_classes: continue
                
                cx, cy, w, h = vals[1:5]
                conf = vals[5] if len(vals) > 5 else 1.0
                
                # 转换回像素坐标
                px, py, bw, bh = cx*W, cy*H, w*W, h*H
                
                # 1. 投影计算 (得到绝对海拔 Z)
                world_x, world_z_abs = project_adaptive(px, py, W, H, meta, global_wall_dist)
                
                # 2. [新增] 楼层计算
                # 直接传入绝对海拔 Z
                floor_name = floor_mgr.get_floor(world_z_abs)
                
                # 计算物体实际高度
                fx = (meta['focal_35mm'] / 36.0) * W
                real_h = (bh / fx) * global_wall_dist
                all_dets.append({
                    "gid": gid,
                    "img": name,
                    "cls": cls,
                    "conf": conf,
                    "cxcywh": (cx, cy, w, h),
                    "pxpywh": (px, py, bw, bh),
                    "x": world_x,
                    "z": world_z_abs,
                    "h": real_h,
                    "img_w": W,  # [新增] 图片宽度
                    "img_h": H,  # [新增] 图片高度
                    "floor": floor_name,
                    # 保存一些元数据方便 debug
                    "meta_alt": meta['alt'],
                    "meta_lrf": meta['lrf'] 
                })
                gid += 1
                
    return all_dets


def merge_boxes_by_id(dets_by_img):
    print("🔄 正在生成合并版数据 (Calculating Union Boxes)...")
    merged_dets_by_img = {}

    for img_name, dets in dets_by_img.items():
        # 1. 按 ID 分组
        id_groups = {}
        for d in dets:
            id_groups.setdefault(d['id'], []).append(d)
        
        merged_list = []
        for uid, group in id_groups.items():
            # 如果该 ID 只有一个框，直接保留
            if len(group) == 1:
                merged_list.append(group[0])
                continue
            
            # === 如果有多个框，执行合并逻辑 ===
            
            # 1. 提取所有框的像素边界
            x1s = [d['pxpywh'][0] - d['pxpywh'][2]/2 for d in group]
            y1s = [d['pxpywh'][1] - d['pxpywh'][3]/2 for d in group]
            x2s = [d['pxpywh'][0] + d['pxpywh'][2]/2 for d in group]
            y2s = [d['pxpywh'][1] + d['pxpywh'][3]/2 for d in group]
            
            # 2. 计算并集 (Union) 的大框坐标
            union_x1 = min(x1s)
            union_y1 = min(y1s)
            union_x2 = max(x2s)
            union_y2 = max(y2s)
            
            # 3. 算出新的像素中心和宽高
            new_bw = union_x2 - union_x1
            new_bh = union_y2 - union_y1
            new_px = union_x1 + new_bw / 2
            new_py = union_y1 + new_bh / 2
            
            # 4. 转换回 YOLO 归一化坐标 (cx, cy, w, h)
            # 必须使用该图片记录的 img_w, img_h
            W = group[0]['img_w']
            H = group[0]['img_h']
            
            new_cx = new_px / W
            new_cy = new_py / H
            new_nw = new_bw / W # normalized width
            new_nh = new_bh / H # normalized height
            
            # 5. 构造新的检测对象 (复制第一个作为模板)
            new_det = group[0].copy()
            new_det['cxcywh'] = (new_cx, new_cy, new_nw, new_nh) # 更新 YOLO 坐标
            new_det['pxpywh'] = (new_px, new_py, new_bw, new_bh) # 更新像素坐标
            new_det['conf'] = max([d['conf'] for d in group])     # 更新置信度 (取最大)
            
            # 注意：投影坐标(x, z) 此时取中心点的投影可能不太准，
            # 但既然合并了，说明是一个物体，保留原来的 x,z 或者重新投影都可以。
            # 这里简单保留模板的 x,z 仅供参考。
            
            merged_list.append(new_det)
            
        merged_dets_by_img[img_name] = merged_list
        
    return merged_dets_by_img

def yolo_project2facade(img_dir, yolo_txt_dir, proj_params, target_classes=None):
    """
    读取 YOLO 标签并进行投影。
    增加 target_classes 参数进行筛选。
    """
    all_dets = []
    gid = 0
    
    # 如果指定了筛选类别
    if target_classes is not None:
        print(f"🎯 仅保留类别 ID: {target_classes}")
    else:
        print("🌐 保留所有类别")

    print("📥 读取 YOLO + 投影到立面坐标...")

    txt_files = glob.glob(os.path.join(yolo_txt_dir, "*.txt"))
    for txt in tqdm(txt_files):
        name = os.path.splitext(os.path.basename(txt))[0]
        img_path = os.path.join(img_dir, name + ".jpg")
        
        # 图片校验，如果图片不存在则跳过（防止尺寸未知）
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as img:
            W, H = img.size

        with open(txt) as f:
            for line in f:
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue

                cls, cx, cy, w, h, conf = parsed
                
                # [新增功能] 类别筛选
                if target_classes is not None and cls not in target_classes:
                    continue

                px, py, bw, bh = yolo_to_pixel(cx, cy, w, h, W, H)
                x, z = project_to_facade(px, py, W, H, proj_params)

                all_dets.append({
                    "gid": gid,
                    "img": name,
                    "cls": cls,
                    "conf": conf,
                    "cxcywh": (cx, cy, w, h),
                    "pxpywh": (px, py, bw, bh),
                    "x": x,
                    "z": z,
                    "h": bh / H * proj_params["facade_distance_m"],
                })
                gid += 1

    print(f"✅ 共读取 {len(all_dets)} 个有效检测")
    return all_dets

def yolo_dedup(all_dets, iou_thresh, height_thresh_m, x_thresh_m=2.0):
    """
    x_thresh_m: 新增参数，水平方向允许的最大合并距离（米）
    """
    print("🧹 基于立面高度 + 水平距离去重...")

    assigned = {}
    clusters = {}
    cid = 0

    for i, a in enumerate(tqdm(all_dets)):
        if a["gid"] in assigned:
            continue

        clusters[cid] = [a]
        assigned[a["gid"]] = cid

        for b in all_dets[i + 1:]:
            if b["gid"] in assigned:
                continue
            if a["cls"] != b["cls"]:
                continue

            # ===================== 新增修复逻辑 =====================
            
            # 1. 【同图互斥检查】
            # 如果两个框来自同一张图片，且像素 IoU 很小（没有重叠），
            # 那么它们绝对是两个不同的物体，禁止合并！
            if a["img"] == b["img"]:
                pixel_iou = compute_iou_2d(a["cxcywh"], b["cxcywh"])
                if pixel_iou < 0.1: # 阈值很低，只要不重叠就是不同物体
                    continue

            # 2. 【水平距离检查】(简单的 X 轴过滤)
            # 计算投影后的水平 X 距离
            # 注意：这假设无人机主要是垂直飞行(做电梯运动)，或者 X 轴相对位置变化不大
            x_diff = abs(a["x"] - b["x"])
            if x_diff > x_thresh_m:
                continue
            
            # =======================================================

            # 原有的高度检查逻辑
            a_min, a_max = a["z"] - a["h"]/2, a["z"] + a["h"]/2
            b_min, b_max = b["z"] - b["h"]/2, b["z"] + b["h"]/2

            if abs(a["z"] - b["z"]) < height_thresh_m and \
               iou_1d(a_min, a_max, b_min, b_max) > iou_thresh:
                clusters[cid].append(b)
                assigned[b["gid"]] = cid

        cid += 1

    print(f"✅ 形成 {len(clusters)} 个唯一物体 ID")
    for d in all_dets:
        d["id"] = assigned[d["gid"]]
    return all_dets


def group_dets_by_image(all_dets):
    """
    [关键修复] 将检测结果按图片文件名分组，以便写入和可视化
    """
    by_img = {}
    for d in all_dets:
        by_img.setdefault(d["img"], []).append(d)
    return by_img

def yolo_dedup_write(by_img, dedup_label_dir):
    print("✍️ 写回 YOLO txt（带 id）...")
    for img_name, dets in by_img.items():
        with open(os.path.join(dedup_label_dir, img_name + ".txt"), "w") as f:
            for d in dets:
                cx, cy, w, h = d["cxcywh"]
                # 写入格式: class cx cy w h conf id
                f.write(f"{d['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {d['conf']:.4f} {d['id']}\n")

def dedup_vis_colored(dets_by_img, img_dir, save_dir, class_names=None, font_size=20):
    print(f"🖼️ 正在生成可视化 (保存至 {save_dir})...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载字体
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except:
        font = ImageFont.load_default()

    for img_name, dets in tqdm(dets_by_img.items()):
        img_path = os.path.join(img_dir, img_name + ".jpg") # 兼容 .JPG
        if not os.path.exists(img_path):
             img_path = os.path.join(img_dir, img_name + ".JPG")
        if not os.path.exists(img_path): continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for d in dets:
            # 1. 获取颜色
            cls_idx = int(d['cls'])
            color = COLOR_PALETTE[cls_idx % len(COLOR_PALETTE)]
            
            # 2. 画框
            px, py, bw, bh = d["pxpywh"]
            x1, y1 = px - bw/2, py - bh/2
            x2, y2 = px + bw/2, py + bh/2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 3. 准备标签文字
            if class_names and 0 <= cls_idx < len(class_names):
                cls_str = class_names[cls_idx]
            else:
                cls_str = str(cls_idx)
            
            label = f"{cls_str}|ID:{d['id']}"
            
            # 4. 画文字背景 (自适应大小)
            text_bbox = font.getbbox(label) 
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # 背景色跟随框的颜色，文字根据亮度自动黑白
            draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
            
            # 简单判断亮度决定文字颜色
            luminance = (0.299*color[0] + 0.587*color[1] + 0.114*color[2])
            txt_color = (0,0,0) if luminance > 128 else (255,255,255)
            
            draw.text((x1 + 3, y1 - text_h - 6), label, fill=txt_color, font=font)

        img.save(os.path.join(save_dir, img_name + ".jpg"))

def dedup_vis(by_img, img_dir, vis_all_dir, vis_by_id_dir, class_names=None, font_size=20):
    """
    [新增功能] 
    1. 支持 class_names 映射
    2. 支持字体大小设置
    """
    print("🖼️ 生成可视化（增强版）...")
    
    # 尝试加载字体
    try:
        # Windows 常用字体路径
        sys_font = "arial.ttf"
        if platform.system() == "Windows":
            sys_font = "arial.ttf" 
        elif platform.system() == "Linux":
            sys_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        
        font = ImageFont.truetype(sys_font, size=font_size)
    except IOError:
        print("⚠️ 无法加载系统字体，使用默认字体（大小不可调）")
        font = ImageFont.load_default()

    for img_name, dets in tqdm(by_img.items()):
        img_path = os.path.join(img_dir, img_name + ".jpg")
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for d in dets:
            px, py, bw, bh = d["pxpywh"]
            x1, y1 = px - bw/2, py - bh/2
            x2, y2 = px + bw/2, py + bh/2

            # 获取显示名称
            cls_idx = d['cls']
            if class_names and 0 <= cls_idx < len(class_names):
                cls_str = class_names[cls_idx]
            else:
                cls_str = str(cls_idx)
            
            label_text = f"{cls_str}|ID:{d['id']}"

            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # 绘制文字背景框（增强可读性）
            # getbbox 返回 (left, top, right, bottom)
            text_bbox = font.getbbox(label_text) 
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            # 文字背景位置
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill="red")
            # 文字
            draw.text((x1 + 2, y1 - text_h - 4), label_text, fill="white", font=font)

        # 保存总览图
        vis_path = os.path.join(vis_all_dir, img_name + ".jpg")
        img.save(vis_path)

        # 保存分 ID 图 (Cropped or Full image copied)
        # 这里维持原逻辑：把整张图复制到对应ID文件夹
        for d in dets:
            id_dir = os.path.join(vis_by_id_dir, f"id_{d['id']:03d}")
            os.makedirs(id_dir, exist_ok=True)
            shutil.copy(vis_path, os.path.join(id_dir, img_name + ".jpg"))


def export_debug_json(all_dets, output_path, class_names=None):
    print("📊 正在生成调试用 JSON 报告...")
    
    # 1. 按 ID 分组
    groups = {}
    for d in all_dets:
        uid = int(d["id"])  # 确保是 python int
        groups.setdefault(uid, []).append(d)
    
    # 2. 构造输出字典
    json_output = {}
    
    # 按 ID 排序方便查看
    for uid in sorted(groups.keys()):
        dets = groups[uid]
        
        # 获取类别名称
        cls_idx = int(dets[0]['cls'])
        cls_name = class_names[cls_idx] if class_names and 0 <= cls_idx < len(class_names) else str(cls_idx)
        
        # 计算该组的统计信息（方便快速排查）
        z_values = [d['z'] for d in dets]
        x_values = [d['x'] for d in dets]
        avg_z = sum(z_values) / len(z_values)
        avg_x = sum(x_values) / len(x_values)
        
        # 构造该 ID 的详细记录列表
        records = []
        for d in dets:
            records.append({
                "img_name": d['img'],       # 原始文件名
                "gid": int(d['gid']),       # 原始读取顺序的全局ID
                "conf": float(f"{d['conf']:.4f}"),
                # 原始 YOLO 坐标 (cx, cy, w, h)
                "raw_bbox": [float(f"{x:.4f}") for x in d['cxcywh']], 
                # 投影后的世界坐标
                "proj_x": float(f"{d['x']:.4f}"),  # 水平位置
                "proj_z": float(f"{d['z']:.4f}"),  # 垂直高度
                "proj_h": float(f"{d['h']:.4f}"),  # 物体实际高度
                "floor": d['floor'],

            })
            
        json_output[f"ID_{uid:03d}"] = {
            "class": cls_name,
            "count": len(records),
            "stats": {
                "avg_z": float(f"{avg_z:.4f}"),
                "avg_x": float(f"{avg_x:.4f}"),
                "min_z": float(f"{min(z_values):.4f}"),
                "max_z": float(f"{max(z_values):.4f}")
            },
            "instances": records
        }

    # 3. 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 调试报告已保存至: {output_path}")


def analyze_and_vis_conflicts(dets_by_img, img_dir, output_dir, class_names=None, vis_font_size=20):
    print("🕵️ 正在分析同图 ID 冲突并生成合并预览...")
    
    # 路径准备
    conflict_vis_dir = os.path.join(output_dir, "vis_conflicts_audit")
    os.makedirs(conflict_vis_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "intra_image_conflicts_report.txt")
    
    # 字体加载
    try:
        font = ImageFont.truetype("arial.ttf", size=vis_font_size) # Windows
    except:
        font = ImageFont.load_default()

    conflict_count = 0
    
    with open(report_path, "w", encoding="utf-8") as f_rpt:
        f_rpt.write("Image_Name, ID, Class, Count, Merge_Suggestion\n")
        
        for img_name, dets in tqdm(dets_by_img.items()):
            # 1. 在当前图片内，按 ID 分组
            id_groups = {}
            for d in dets:
                id_groups.setdefault(d['id'], []).append(d)
            
            # 2. 检查是否有 ID 的数量 > 1
            has_conflict = False
            img_conflicts = [] # 记录当前图的冲突组
            
            for uid, group in id_groups.items():
                if len(group) > 1:
                    has_conflict = True
                    conflict_count += 1
                    
                    # 获取类别名
                    cls_idx = group[0]['cls']
                    cls_str = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                    
                    # 写入报告
                    # 计算这一组在图片上的最大跨度，帮助判断是否应该合并
                    xs = [d['pxpywh'][0] for d in group]
                    span_px = max(xs) - min(xs)
                    suggestion = f"Span {int(span_px)}px"
                    f_rpt.write(f"{img_name}, {uid}, {cls_str}, {len(group)}, {suggestion}\n")
                    
                    img_conflicts.append(group)

            # 3. 如果有冲突，生成专门的“审计图”
            if has_conflict:
                img_path = os.path.join(img_dir, img_name + ".jpg") # 假设是 jpg，需注意扩展名
                if not os.path.exists(img_path):
                     # 尝试 .JPG
                    img_path = os.path.join(img_dir, img_name + ".JPG")
                
                if not os.path.exists(img_path): continue
                
                with Image.open(img_path).convert("RGB") as img:
                    draw = ImageDraw.Draw(img)
                    
                    # A. 先画所有的常规框 (按类别着色)
                    for d in dets:
                        cls_color = get_class_color(d['cls'])
                        px, py, bw, bh = d['pxpywh']
                        x1, y1 = px - bw/2, py - bh/2
                        x2, y2 = px + bw/2, py + bh/2
                        
                        # 画实线小框
                        draw.rectangle([x1, y1, x2, y2], outline=cls_color, width=3)
                        
                        # 标签
                        cls_idx = d['cls']
                        cls_str = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                        label = f"{cls_str}|{d['id']}"
                        
                        # 标签背景
                        text_bbox = font.getbbox(label)
                        tw, th = text_bbox[2]-text_bbox[0], text_bbox[3]-text_bbox[1]
                        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill=cls_color)
                        txt_color = get_contrasting_text_color(cls_color)
                        draw.text((x1 + 2, y1 - th - 4), label, fill=txt_color, font=font)

                    # B. 再画“合并预览框” (Union Box) - 只针对有冲突的组
                    for group in img_conflicts:
                        # 计算 Union Box 坐标
                        all_x1 = [d['pxpywh'][0] - d['pxpywh'][2]/2 for d in group]
                        all_y1 = [d['pxpywh'][1] - d['pxpywh'][3]/2 for d in group]
                        all_x2 = [d['pxpywh'][0] + d['pxpywh'][2]/2 for d in group]
                        all_y2 = [d['pxpywh'][1] + d['pxpywh'][3]/2 for d in group]
                        
                        ux1, uy1 = min(all_x1), min(all_y1)
                        ux2, uy2 = max(all_x2), max(all_y2)
                        
                        # 画一个醒目的白色大框包围它们
                        # 模拟虚线效果不好做，直接用粗白线 + 内部无填充
                        draw.rectangle([ux1-5, uy1-5, ux2+5, uy2+5], outline="white", width=4)
                        
                        # 在大框顶部写上提示
                        merge_label = f"MERGE PREVIEW: ID {group[0]['id']} (x{len(group)})"
                        draw.text((ux1, uy1 - 30), merge_label, fill="white", font=font, stroke_width=2, stroke_fill="black")

                    # 保存图片到 vis_conflicts_audit 文件夹
                    img.save(os.path.join(conflict_vis_dir, img_name + "_audit.jpg"))

    print(f"✅ 冲突分析完成！发现 {conflict_count} 处潜在合并。")
    print(f"📄 报告已保存: {report_path}")
    print(f"🖼️ 可视化已保存: {conflict_vis_dir}")

def yolo_dedup_pipeline(img_dir, yolo_txt_dir, output_dir, floor_param,
                        iou_thresh, height_thresh_m, x_thresh_m=2.0,
                        target_classes=None, class_names_path=None, vis_font_size=24):
    t1 = time.time()

    # 路径准备
    dedup_label_dir = os.path.join(output_dir, "labels_dedup")
    vis_all_dir = os.path.join(output_dir, "vis_all")
    vis_by_id_dir = os.path.join(output_dir, "vis_by_id")
    proj_info_path = os.path.join(output_dir, "project_info.json")

    dedup_label_dir_fuse = os.path.join(output_dir, "labels_dedup_fuse")
    vis_all_dir_fuse = os.path.join(output_dir, "vis_all_fuse")
    vis_by_id_dir_fuse = os.path.join(output_dir, "vis_by_id_fuse")
    group_info_path = os.path.join(output_dir, "labels_group_info.json")
    
    os.makedirs(dedup_label_dir, exist_ok=True)
    os.makedirs(vis_all_dir, exist_ok=True)
    os.makedirs(vis_by_id_dir, exist_ok=True)
    os.makedirs(dedup_label_dir_fuse, exist_ok=True)
    os.makedirs(vis_all_dir_fuse, exist_ok=True)
    os.makedirs(vis_by_id_dir_fuse, exist_ok=True)

    # 0. 加载类别名称
    class_names = load_class_names(class_names_path)

    # 1. 读取并筛选
    all_dets = yolo_project2facade_adaptive(img_dir, yolo_txt_dir, target_classes, floor_param)
    
    # 2. 去重
    all_dets_with_id = yolo_dedup(all_dets, iou_thresh, height_thresh_m, x_thresh_m)
    
    # 3. 按图片分组 (关键修复)
    dets_by_img = group_dets_by_image(all_dets_with_id)
    dets_by_img_fuse = merge_boxes_by_id(dets_by_img)

    analyze_and_vis_conflicts(
            dets_by_img, 
            img_dir, 
            output_dir, 
            class_names=class_names, # 记得传入 class_names
            vis_font_size=vis_font_size
        )

    # 4. 写入
    yolo_dedup_write(dets_by_img, dedup_label_dir)
    yolo_dedup_write(dets_by_img_fuse, dedup_label_dir_fuse)

    # 5. 可视化
    dedup_vis(dets_by_img, img_dir, vis_all_dir, vis_by_id_dir, 
              class_names=class_names, font_size=vis_font_size)
    dedup_vis(dets_by_img_fuse, img_dir, vis_all_dir_fuse, vis_by_id_dir_fuse, 
              class_names=class_names, font_size=vis_font_size)
    
    # 6. 生成调试用 JSON 报告
    export_debug_json(all_dets_with_id, group_info_path, class_names)

    # 7. 导出投影详情 JSON
    export_projection_details_json(all_dets_with_id, proj_info_path)

    t2 = time.time()

    print(f"🎉 完成：YOLO 投影去重 + ID 审计流水线, 耗时 {t2-t1:.2f}s")



if __name__ == "__main__":
    # ===================== 路径配置 =====================
    image_root = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views"
    yolo_root =  r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views_infer"
    output_root = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views_infer_dedup"
    classes_txt_path = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\class.txt" 
    views_list = os.listdir(image_root)

    # ===================== 参数 =====================
    iou_thresh = 0.5
    height_thresh_m = 0.3
    x_thresh_m = 1.5
    
    # [新增] 筛选配置
    # 如果想保留所有类别，设为 None
    # 如果只想保留类别 0 和 2，设为 [0, 2]
    # filter_classes = None
    filter_classes = [0, 4]

    # [新增] 可视化字体大小
    vis_font_size = 30

    floor_param = {
        'base_height':22500,
        'final height':123800,
        'normal floor height':3150,
        'podium heights': [6000, 5000, 4500, 5500],
        'top heights': [6650],
        'podium names': ['LG', 'G', '1', '2'],
        'top names': ['ROOF'],
        'normal height numbers': 23,
        'normal height number list': [3, 25],
        'special heights': {
            '4': 3450,
            '11': 3450,
            '18': 3450,
            '23': 3450,
        }
    }


    for view_name in views_list:

        image_dir = os.path.join(image_root, view_name)
        yolo_dir = os.path.join(yolo_root, view_name, "labels")
        output_dir = os.path.join(output_root, view_name)

        yolo_dedup_pipeline(
            img_dir=image_dir, 
            yolo_txt_dir=yolo_dir, 
            output_dir=output_dir,
            iou_thresh=iou_thresh, 
            height_thresh_m=height_thresh_m,
            target_classes=filter_classes,
            class_names_path=classes_txt_path,
            vis_font_size=vis_font_size,
            x_thresh_m=x_thresh_m,
            floor_param=floor_param,
        )