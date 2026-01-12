import os
import re
import math
import csv
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm

import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# =========================
# 基础配置与 Helper
# =========================

def get_exif(img_path: str) -> dict:
    # 请替换为你真实的 pyexif 调用
    # 这里仅做演示
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()

def parse_float(val) -> Optional[float]:
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip().replace("deg", "").strip()
    try:
        return float(s)
    except:
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None

def dms_to_dd(dms_str: str) -> Optional[float]:
    if dms_str is None: return None
    s = str(dms_str).strip()
    try:
        return float(s)
    except:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)\"?\s*([NSEW])", s, re.IGNORECASE)
    if not m: return None
    deg, minute, sec, hemi = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4).upper()
    dd = deg + minute/60.0 + sec/3600.0
    return -dd if hemi in ("S", "W") else dd

def parse_gps_from_exif(exif: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = dms_to_dd(exif.get("GPSLatitude")), dms_to_dd(exif.get("GPSLongitude"))
    if (lat is None or lon is None) and exif.get("GPSPosition"):
        parts = [p.strip() for p in str(exif.get("GPSPosition")).split(",")]
        if len(parts) >= 2:
            lat = lat or dms_to_dd(parts[0])
            lon = lon or dms_to_dd(parts[1])
    return lat, lon

# ... (pick_first_image, pick_last_image, pick_middle_image 保持不变) ...
def pick_first_image(folder: Path) -> Optional[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs[0] if imgs else None

def pick_last_image(folder: Path) -> Optional[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs[-1] if imgs else None


def pick_middle_image(folder: Path) -> Optional[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs[len(imgs) // 2] if imgs else None
# =========================
# 几何与颜色逻辑 (8色/渐变)
# =========================
def forward_geodesic(lat_deg, lon_deg, bearing_deg, distance_m) -> Tuple[float, float]:
    R = 6378137.0
    lat1, lon1, brng = map(math.radians, [lat_deg, lon_deg, bearing_deg])
    d = distance_m / R
    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d)*math.cos(lat1), math.cos(d)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

def hex_to_rgb(hex_color: str):
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*map(int, rgb))

def blend_colors(c1_hex, c2_hex, ratio):
    c1, c2 = hex_to_rgb(c1_hex), hex_to_rgb(c2_hex)
    return rgb_to_hex([c1[i]*(1-ratio) + c2[i]*ratio for i in range(3)])

def get_dynamic_bearing_color(bearing_deg: float) -> str:
    COLORS = {0: "#FFD700", 90: "#FF0000", 180: "#008000", 270: "#0000FF", 360: "#FFD700"}
    b = bearing_deg % 360.0
    quadrant = int(b // 90)
    start_a, end_a = quadrant * 90, (quadrant + 1) * 90
    return blend_colors(COLORS[start_a], COLORS[end_a], (b - start_a) / 90.0)

def get_cardinal_direction(yaw_deg: float) -> str:
    dirs = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    return dirs[int((yaw_deg + 22.5) % 360 / 45)]

def make_rotated_triangle_icon(color: str, bearing_deg: float) -> folium.DivIcon:
    html = f"""<div style="width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-bottom:16px solid {color};transform:rotate({bearing_deg:.2f}deg);transform-origin:50% 60%;"></div>"""
    return folium.DivIcon(html=html)

def parse_fov_lrf(exif):
    return parse_float(exif.get("FOV")), parse_float(exif.get("LRFTargetLat")), parse_float(exif.get("LRFTargetLon")), parse_float(exif.get("LRFTargetDistance"))

# =========================
# 截图核心逻辑 (新增)
# =========================
def batch_screenshot_views(points: List[Dict], output_dir: str, arrow_len_m: float, gap_m: float):
    """
    修正版：
    总览图 (Overview) 的缩放范围只聚焦于无人机位置，忽略可能过长的 LRF 红线，
    确保箭头在总览图中清晰可见。
    """
    print(f"\n[INFO] 开始截图任务 (总览图 + {len(points)} 张特写)...")
    
    shots_dir = Path(output_dir)
    shots_dir.mkdir(parents=True, exist_ok=True)
        
    temp_html = shots_dir / "temp_view.html"

    # Selenium 配置
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--window-size=1200,1200") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--ignore-certificate-errors")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # Helper: 添加 OSM 底图
    def add_osm_tile(m_obj):
        folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="OpenStreetMap",
            max_zoom=25,
            max_native_zoom=19,
            control=False
        ).add_to(m_obj)

    # Helper: 在地图上画一个点
    def draw_arrow_on_map(m_obj, p, include_fov=True):
        lat, lon, yaw = p["lat"], p["lon"], p["yaw"]
        color = get_dynamic_bearing_color(yaw)
        
        # 这里的 points_for_bound 包含了箭头和红线的所有端点
        points_for_bound = [(lat, lon)]

        # 1. 箭头杆
        line_len = max(0.5, arrow_len_m - gap_m)
        lat_end, lon_end = forward_geodesic(lat, lon, yaw, line_len)
        folium.PolyLine([(lat, lon), (lat_end, lon_end)], weight=4, color=color, opacity=1).add_to(m_obj)
        
        # 2. 箭头头
        lat_tip, lon_tip = forward_geodesic(lat, lon, yaw, arrow_len_m)
        folium.Marker((lat_tip, lon_tip), icon=make_rotated_triangle_icon(color, yaw)).add_to(m_obj)
        points_for_bound.append((lat_tip, lon_tip))

        # 3. LRF 视场线 (红线)
        if include_fov:
            tlat, tlon, tdist, fov = p["tlat"], p["tlon"], p["tdist"], p["fov_deg"]
            if all(v is not None for v in [tlat, tlon, tdist, fov]) and tdist > 0:
                half_w = tdist * math.tan(math.radians(fov / 2.0))
                l_lat, l_lon = forward_geodesic(tlat, tlon, (yaw - 90) % 360, half_w)
                r_lat, r_lon = forward_geodesic(tlat, tlon, (yaw + 90) % 360, half_w)
                
                folium.PolyLine([(l_lat, l_lon), (r_lat, r_lon)], weight=2, color="gray", opacity=0.8).add_to(m_obj)
                points_for_bound.extend([(l_lat, l_lon), (r_lat, r_lon)])
        
        return points_for_bound

    try:
        # ==========================================
        # Part 1: 生成并截取“总览图” (Overview)
        # ==========================================
        print("  [0/N] Generating Overview Map...")
        
        avg_lat = sum(p['lat'] for p in points) / len(points)
        avg_lon = sum(p['lon'] for p in points) / len(points)
        
        m_all = folium.Map(location=[avg_lat, avg_lon], zoom_start=20, tiles=None)
        add_osm_tile(m_all) 

        all_bounds = []
        for p in points:
            # 1. 画箭头和红线 (Visual)
            # p_bounds 包含了红线远端，但我们不用它来 fit_bounds
            _ = draw_arrow_on_map(m_all, p, include_fov=True) 
            
            # 2. 【关键修改】只收集无人机自身坐标用于 fit_bounds
            # 这样地图就会聚焦在飞行路线上，而不会被偶尔出现的超长红线拉远视角
            all_bounds.append((p['lat'], p['lon']))
            
            # # 也可以把箭头尖端加进去，保证箭头不被切掉，但绝不加红线端点
            lat_tip, lon_tip = forward_geodesic(p['lat'], p['lon'], p['yaw'], arrow_len_m)
            all_bounds.append((lat_tip, lon_tip))

            folium.Marker(
                (p['lat'], p['lon']),
                icon=folium.DivIcon(html=f'<div style="font-size:10px;font-weight:bold;color:#000;text-shadow: 1px 1px 0 #fff;">{p["vid"]}</div>')
            ).add_to(m_all)

        if all_bounds:
            # Padding 适当增加，保证周围有空隙
            m_all.fit_bounds(all_bounds, padding=(10, 10))

        m_all.save(str(temp_html))
        driver.get(f"file:///{temp_html.absolute()}")
        time.sleep(2.0) 
        driver.save_screenshot(str(shots_dir / "_Overview_All_Arrows.png"))
        print(f"  [OK] Overview Saved: _Overview_All_Arrows.png")


        # ==========================================
        # Part 2: 生成并截取“单点特写” (Individual)
        # ==========================================
        for i, p in enumerate(points):
            m_single = folium.Map(location=[p["lat"], p["lon"]], zoom_start=22, tiles=None)
            add_osm_tile(m_single) 

            # 特写图需要包含红线范围，所以这里使用 draw_arrow_on_map 返回的完整 bounds
            bounds_points = draw_arrow_on_map(m_single, p, include_fov=True)
            
            # 使用您满意的 padding
            m_single.fit_bounds(bounds_points, padding=(100, 100)) 
            
            m_single.save(str(temp_html))
            driver.get(f"file:///{temp_html.absolute()}")
            
            time.sleep(1.0)
            
            out_name = f"{p['vid']}_{p['folder']}_{p['cardinal_dir']}.png"
            driver.save_screenshot(str(shots_dir / out_name))
            
            print(f"  [{i+1}/{len(points)}] Saved: {out_name}")

    except Exception as e:
        print(f"[ERROR] 截图中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()
        if temp_html.exists():
            os.remove(temp_html)
    
    print(f"[OK] 所有截图任务完成: {shots_dir}")

# =========================
# Main Logic
# =========================
def process_views_data(
    root_dir: str,
    output_folder: str = "output_results",
    pick_method: str = "middle",
    arrow_len_m: float = 3.0,
    gap_m: float = 1.2,
    yaw_offset_deg: float = 0.0,
):
    root = Path(root_dir)
    out_path = Path(output_folder)
    out_path.mkdir(exist_ok=True)

    html_file = out_path / "views_map.html"
    csv_file = out_path / "views_direction.csv"
    screenshots_dir = out_path / "screenshots"


    folders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    points = []

    if pick_method == "first":
        pick_func = pick_first_image
    elif pick_method == "last":
        pick_func = pick_last_image
    elif pick_method == "middle":
        pick_func = pick_middle_image
    else:
        raise ValueError(f"Unknown pick_method: {pick_method}")

    print(f"[INFO] Parsing {len(folders)} folders EXIF data...")
    for i, folder in enumerate(tqdm(folders), 1):
        img = pick_func(folder) # 这里默认用第一张
        if not img: continue
        
        exif = get_exif(str(img))
        lat, lon = parse_gps_from_exif(exif)
        if not lat: continue
        
        yaw = (parse_float(exif.get("GimbalYawDegree") or exif.get("FlightYawDegree") or 0) + yaw_offset_deg) % 360
        fov, tlat, tlon, tdist = parse_fov_lrf(exif)
        
        points.append({
            "vid": f"V{i}", "folder": folder.name, "img": img.name,
            "lat": lat, "lon": lon, "yaw": yaw,
            "cardinal_dir": get_cardinal_direction(yaw),
            "fov_deg": fov, "tlat": tlat, "tlon": tlon, "tdist": tdist
        })

    # 1. 导出 CSV
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["VID", "View", "Image", "Yaw", "Direction", "Lat", "Lon"])
        writer.writeheader()
        for p in points:
            writer.writerow({"VID": p["vid"], "View": p["folder"], "Image": p["img"], 
                             "Yaw": f"{p['yaw']:.2f}", "Direction": p["cardinal_dir"], 
                             "Lat": p["lat"], "Lon": p["lon"]})
    print(f"[OK] CSV Saved: {csv_file}")

    # 2. 生成总览地图 (Views Map)
    center = [sum(p["lat"] for p in points)/len(points), sum(p["lon"] for p in points)/len(points)]
    m = folium.Map(location=center, zoom_start=20, tiles=None)
    folium.TileLayer("https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
                     attr="Esri", name="Esri Clarity", max_zoom=24, max_native_zoom=19).add_to(m)
    
    for p in points:
        color = get_dynamic_bearing_color(p["yaw"])
        # 画线
        le_lat, le_lon = forward_geodesic(p["lat"], p["lon"], p["yaw"], max(0.5, arrow_len_m - gap_m))
        folium.PolyLine([(p["lat"], p["lon"]), (le_lat, le_lon)], color=color, weight=4, opacity=0.9).add_to(m)
        # 画头
        lt_lat, lt_lon = forward_geodesic(p["lat"], p["lon"], p["yaw"], arrow_len_m)
        folium.Marker((lt_lat, lt_lon), icon=make_rotated_triangle_icon(color, p["yaw"])).add_to(m)
        # 画标签
        folium.Marker((p["lat"], p["lon"]), icon=folium.DivIcon(html=f'<div style="font-size:10px;background:rgba(255,255,255,0.7);padding:1px;">{p["vid"]}</div>')).add_to(m)

    m.save(str(html_file))
    print(f"[OK] Map Saved: {html_file}")

    # 3. [新增] 批量截图
    # 将会为每个箭头生成一个特写图
    batch_screenshot_views(points, str(screenshots_dir), arrow_len_m, gap_m)



if __name__ == "__main__":
    ROOT_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\visible_views"
    output_html = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs\visible_views_map.html'
    output_csv = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs\visible_views_map.csv'
    screenshots_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs\visible_views_map.csv'
    output_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\docs'
    process_views_data(
        ROOT_DIR,
        output_dir,
        arrow_len_m=3.0,
        gap_m=1.2,
        yaw_offset_deg=0.0,
    )