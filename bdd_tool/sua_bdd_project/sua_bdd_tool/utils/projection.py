import re
import math

def get_exif(img_path) :
    # 请替换为你真实的 pyexif 调用
    # 这里仅做演示
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()

def parse_float(val):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace("deg", "").strip()
    try:
        return float(s)
    except:
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None

def parse_int(val):
    val_float = parse_float(val)
    return int(val_float) if val_float is not None else None

def convert_coordinate(lat, lon):
    if lat is not None and lon is not None:
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        lat_abs = abs(lat)
        lon_abs = abs(lon)
        return f"{lat_abs:.6f}{lat_dir}, {lon_abs:.6f}{lon_dir}"
    else:
        return "N/A"

def calculate_gsd(distance_mm, focal_length_mm, sensor_width_mm, image_width_pix):
    """
    计算地面采样距离 (GSD): 每个像素代表的物理世界毫米数
    Formula: GSD = (Distance * SensorWidth) / (FocalLength * ImageWidth)
    """
    if any(v is None or v == 0 for v in [distance_mm, focal_length_mm, sensor_width_mm, image_width_pix]):
        return None
        
    try:
        gsd = (float(distance_mm) * float(sensor_width_mm)) / (float(focal_length_mm) * float(image_width_pix))
        return gsd # unit: mm/pixel
    except ZeroDivisionError:
        return None

def calculate_facade_gsd(distance_m, focal_length, pixel_size_um=None, cos_theta = 1.0):
    gsd_result = (distance_m * (pixel_size_um / 1000)) / (focal_length * cos_theta)*1000
    return gsd_result

def pixel_to_physical(pix_value, gsd):
    """
    将像素值转换为厘米 (cm)
    """
    if pix_value is None or gsd is None:
        return None
    
    mm_value = pix_value * gsd
    cm_value = mm_value / 10.0
    return cm_value


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


def dms_to_dd(dms_str):
    """
    解析 DMS 格式 (e.g. '22 deg 18' 35.81" N') 为十进制度数
    支持小数度分秒，支持无引号结尾
    """
    if dms_str is None:
        return None
    
    s = str(dms_str).strip()
    
    # 1. 尝试直接转换为数字 (应对已经是小数的情况)
    try:
        return float(s)
    except ValueError:
        pass
    
    # 2. 正则解析 DMS 格式
    # 匹配: 数字(可含小数) deg 数字(可含小数)' 数字(可含小数)"? 方向
    pattern = r"(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)\"?\s*([NSEW])"
    match = re.search(pattern, s, re.IGNORECASE)
    
    if not match:
        return None
        
    deg = float(match.group(1))
    minute = float(match.group(2))
    sec = float(match.group(3))
    hemi = match.group(4).upper()
    
    # 转换公式
    dd = deg + minute/60.0 + sec/3600.0
    
    # 南纬(S)和西经(W)设为负数
    return -dd if hemi in ("S", "W") else dd

def parse_gps_from_exif(exif):
    """
    从 EXIF 字典中提取经纬度
    """
    # 优先尝试直接读取独立的 Latitude/Longitude 字段
    lat = dms_to_dd(exif.get("GPSLatitude"))
    lon = dms_to_dd(exif.get("GPSLongitude"))
    
    # 如果没读到，尝试解析 GPSPosition 组合字段
    if (lat is None or lon is None) and exif.get("GPSPosition"):
        parts = [p.strip() for p in str(exif.get("GPSPosition")).split(",")]
        if len(parts) >= 2:
            lat = lat or dms_to_dd(parts[0])
            lon = lon or dms_to_dd(parts[1])
            
    return lat, lon


def get_cardinal_direction(yaw_deg: float) -> str:
    dirs = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    return dirs[int((yaw_deg + 22.5) % 360 / 45)]

def forward_geodesic(lat_deg, lon_deg, bearing_deg, distance_m):
    R = 6378137.0
    lat1, lon1, brng = map(math.radians, [lat_deg, lon_deg, bearing_deg])
    d = distance_m / R
    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d)*math.cos(lat1), math.cos(d)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def safe_float(value):
    """
    处理 '6.7 mm', '4.67 m', None, 0 等各种情况转 float
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    
    val_str = str(value).strip()
    try:
        return float(val_str)
    except ValueError:
        # 提取数字部分 (支持 "6.7 mm", "+93.9 m")
        match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
        if match:
            return float(match.group())
        return 0.0

def extract_rdf_description_bytes(jpg_path):
    """
    从 JPG 文件中提取第一个 <rdf:Description ... </rdf:Description> 区块（bytes）。
    """
    _RDF_START = b"<rdf:Description "
    _RDF_END = b"</rdf:Description>"
    _ATTR_RE = re.compile(r'(?P<key>[A-Za-z0-9_:\-]+)\s*=\s*"(?P<val>[^"]*)"')

    data = jpg_path.read_bytes()
    s = data.find(_RDF_START)
    if s < 0:
        return None
    e = data.find(_RDF_END, s)
    if e < 0:
        return None
    e = e + len(_RDF_END)
    return data[s:e]

def parse_dji_xmp(jpg_path):
    """
    解析 DJI Matrice 4T 写入 JPG 的 XMP 扩展元数据（drone-dji:*）。
    返回 dict：key -> value（字符串）
    """
    from pathlib import Path
    _DJI_ATTR_RE = re.compile(r'drone-dji:(?P<key>[A-Za-z0-9_]+)\s*=\s*"(?P<val>[^"]*)"')
    p = Path(jpg_path)
    block = extract_rdf_description_bytes(p)
    if not block:
        return {}

    # 尽量用 UTF-8/ASCII 解码（XMP 通常是 UTF-8）
    text = block.decode("utf-8", errors="ignore")

    out = {}
    for m in _DJI_ATTR_RE.finditer(text):
        out[m.group("key")] = m.group("val")
    return out

if __name__ == '__main__':
    pass
    get_exif(r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\visible\DJI_20251216143254_0001_V.JPG')