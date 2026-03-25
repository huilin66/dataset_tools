import os
import re
import json
import math
import subprocess
from tqdm import tqdm
from typing import Dict, List

import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Transformer


# =========================
# 参数（可调）
# =========================

IMAGE_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views\V21"
OUTPUT_GEOJSON = "projected_views_dedup.geojson"

MAX_DIST_M = 1.5       # 线段最小距离（米） -> 判定重叠
MAX_YAW_DIFF = 5.0     # 朝向差（度）
MAX_LEN_DIFF = 2.0     # 投影长度差（米）

USE_LRF = True         # 优先使用 LRF 距离
DEFAULT_FOV = 73.7     # deg


# =========================
# 工具函数
# =========================

def run_exiftool(img_path: str) -> Dict:
    cmd = ["exiftool", "-json", img_path]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return json.loads(out)[0]


def dms_to_deg(dms: str) -> float:
    """
    Convert DJI-style DMS string to decimal degrees.
    Example: '22 deg 18\' 35.81" N'
    """
    nums = list(map(float, re.findall(r"[\d.]+", dms)))
    if len(nums) < 3:
        raise ValueError(f"Invalid DMS format: {dms}")

    deg, minute, sec = nums[:3]

    sign = -1 if ("S" in dms or "W" in dms) else 1
    return sign * (deg + minute / 60 + sec / 3600)

def parse_latlon(exif: Dict):
    lat = dms_to_deg(exif["GPSLatitude"])
    lon = dms_to_deg(exif["GPSLongitude"])
    if exif.get("GPSLatitudeRef") == "South":
        lat *= -1
    if exif.get("GPSLongitudeRef") == "West":
        lon *= -1
    return lat, lon


def yaw_from_exif(exif: Dict) -> float:
    return float(exif.get("GimbalYawDegree", exif.get("FlightYawDegree", 0.0)))


def get_projection_distance(exif: Dict) -> float:
    if USE_LRF and "LRFTargetDistance" in exif:
        return float(exif["LRFTargetDistance"])
    return float(exif.get("RelativeAltitude", 20.0))


# =========================
# 投影
# =========================

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def project_view_line(exif: Dict) -> LineString:
    lat, lon = parse_latlon(exif)
    yaw = math.radians(yaw_from_exif(exif))
    dist = get_projection_distance(exif)

    x0, y0 = transformer.transform(lon, lat)

    dx = dist * math.sin(yaw)
    dy = dist * math.cos(yaw)

    x1 = x0 + dx
    y1 = y0 + dy

    lon1, lat1 = transformer_back.transform(x1, y1)

    return LineString([(lon, lat), (lon1, lat1)])


# =========================
# 去重逻辑
# =========================

def is_duplicate(a, b) -> bool:
    if a.geometry.distance(b.geometry) > MAX_DIST_M / 111000:
        return False

    yaw_diff = abs(a["yaw"] - b["yaw"])
    yaw_diff = min(yaw_diff, 360 - yaw_diff)

    if yaw_diff > MAX_YAW_DIFF:
        return False

    if abs(a["length"] - b["length"]) > MAX_LEN_DIFF:
        return False

    return True


# =========================
# 主流程
# =========================

records = []

print("📷 读取并投影照片...")
for fn in tqdm(sorted(os.listdir(IMAGE_DIR))):
    if not fn.lower().endswith(".jpg"):
        continue

    path = os.path.join(IMAGE_DIR, fn)
    exif = run_exiftool(path)


    line = project_view_line(exif)


    records.append({
        "filename": fn,
        "yaw": yaw_from_exif(exif),
        "length": get_projection_distance(exif),
        "geometry": line
    })

print(records[0].keys())
gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

print(f"✅ 共投影 {len(gdf)} 张")


# =========================
# 去重
# =========================

print("🧹 去重中...")
keep = []
used = [False] * len(gdf)

for i in tqdm(range(len(gdf))):
    if used[i]:
        continue

    keep.append(i)
    for j in range(i + 1, len(gdf)):
        if used[j]:
            continue
        if is_duplicate(gdf.iloc[i], gdf.iloc[j]):
            used[j] = True


gdf_dedup = gdf.iloc[keep]

print(f"✅ 去重后剩余 {len(gdf_dedup)} 张")


# =========================
# 输出
# =========================

gdf_dedup.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
print(f"📍 已输出：{OUTPUT_GEOJSON}")
