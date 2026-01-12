import csv
import math
import os
from pathlib import Path
import shutil
import time

import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

from sua_bdd_tool.utils.file_opt import (
    pick_first_image,
    pick_last_image,
    pick_middle_image,
)
from sua_bdd_tool.utils import load_json
from sua_bdd_tool.utils.projection import forward_geodesic
from sua_bdd_tool.utils.visualization import get_dynamic_bearing_color


def make_rotated_triangle_icon(color, bearing_deg):
    html = f"""<div style="width:0;height:0;border-left:8px solid transparent;border-right:8px solid transparent;border-bottom:16px solid {color};transform:rotate({bearing_deg:.2f}deg);transform-origin:50% 60%;"></div>"""
    return folium.DivIcon(html=html)

def batch_screenshot_views(points, output_dir, arrow_len_m, gap_m, view_shot_all, view_shot_each):
    """
    修正版：
    总览图 (Overview) 的缩放范围只聚焦于无人机位置，忽略可能过长的 LRF 红线，
    确保箭头在总览图中清晰可见。
    """
    if not view_shot_all and not view_shot_each:
        return
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


        if view_shot_all:
            print("  [0/N] Generating Overview Map Screenshot...")
            driver.get(f"file:///{temp_html.absolute()}")
            time.sleep(2.0) 
            driver.save_screenshot(str(shots_dir.parent / "views_map_overview.png"))
            print(f"  [OK] Overview Saved: views_map_overview.png")

        if view_shot_each:
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
            print(f'All Screenshots Done in {shots_dir}')
        else:
            shutil.rmtree(shots_dir)
    except Exception as e:
        print(f"[ERROR] 截图中断: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()
        if temp_html.exists():
            os.remove(temp_html)
    
    print(f"[OK] 所有截图任务完成!")


def process_views_data(
    root_dir,
    output_folder,
    exif_path,
    pick_method="middle",
    arrow_len_m=3.0,
    gap_m=1.2,
    view_shot_all=True,
    view_shot_each=False,
):
    root = Path(root_dir)
    out_path = Path(output_folder)
    out_path.mkdir(exist_ok=True)

    html_file = out_path / "views_map.html"
    csv_file = out_path / "views_map.csv"
    screenshots_dir = out_path / "views_map_screenshots"

    exif_db = load_json(exif_path)

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
        img_name = Path(img).name
        img_exif = exif_db.get(img_name)

        points.append({
            "vid": f"V{i}", 
            "folder": img_exif["rel_dir"], 
            "img": img_exif["filename"],
            "lat": img_exif["lat"], 
            "lon": img_exif["lon"], 
            "yaw": img_exif["yaw"],
            "cardinal_dir": img_exif["direction"],
            "fov_deg": img_exif["fov"], 
            "tlat": img_exif["tlat"], 
            "tlon": img_exif["tlon"], 
            "tdist": img_exif["tdist"],
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
    batch_screenshot_views(points, str(screenshots_dir), arrow_len_m, gap_m, view_shot_all, view_shot_each)
