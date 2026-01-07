import os
import re
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import folium



# =========================
# 你已有的函数A：输入图片路径，返回 EXIF dict
# 把这里替换成你真实的函数即可
# =========================
def get_exif(img_path: str) -> dict:
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()


# -------------------------
# Parsing helpers
# -------------------------
def parse_float(val) -> Optional[float]:
    """Parse float from '+6.50', '6.50', '6.50 deg', numeric, etc."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    s = s.replace("deg", "").strip()
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None


def dms_to_dd(dms_str: str) -> Optional[float]:
    """
    Convert "22 deg 18' 35.81\" N" to decimal degrees.
    Also accepts already-decimal string like "22.3100105".
    """
    if dms_str is None:
        return None
    s = str(dms_str).strip()

    # If it is already a decimal number
    try:
        return float(s)
    except Exception:
        pass

    # Match degrees, minutes, seconds, hemisphere
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*deg\s*(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)\"?\s*([NSEW])",
        s,
        re.IGNORECASE,
    )
    if not m:
        return None

    deg = float(m.group(1))
    minute = float(m.group(2))
    sec = float(m.group(3))
    hemi = m.group(4).upper()

    dd = deg + minute / 60.0 + sec / 3600.0
    if hemi in ("S", "W"):
        dd = -dd
    return dd


def parse_gps_from_exif(exif: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Try GPSLatitude/GPSLongitude first, then GPSPosition "lat, lon".
    """
    lat = dms_to_dd(exif.get("GPSLatitude"))
    lon = dms_to_dd(exif.get("GPSLongitude"))

    if (lat is None or lon is None) and exif.get("GPSPosition"):
        gps_pos = str(exif.get("GPSPosition"))
        parts = [p.strip() for p in gps_pos.split(",")]
        if len(parts) >= 2:
            lat2 = dms_to_dd(parts[0])
            lon2 = dms_to_dd(parts[1])
            lat = lat if lat is not None else lat2
            lon = lon if lon is not None else lon2

    return lat, lon


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

# -------------------------
# Geometry helpers
# -------------------------
def forward_geodesic(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """
    Move from (lat, lon) along bearing for distance (meters) on a sphere.
    Good enough for short arrows (10–50m).
    """
    R = 6378137.0  # WGS84 approx
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d = distance_m / R

    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2)
    )
    return (math.degrees(lat2), math.degrees(lon2))


def bearing_to_color(bearing_deg: float) -> str:
    """
    按东南西北分象限上色：
    北(N): 黄, 东(E): 红, 南(S): 绿, 西(W): 蓝
    """
    b = bearing_deg % 360.0
    if (315 <= b < 360) or (0 <= b < 45):
        return "yellow"  # North
    elif 45 <= b < 135:
        return "red"     # East
    elif 135 <= b < 225:
        return "green"   # South
    else:
        return "blue"    # West


def make_rotated_triangle_icon(color: str, bearing_deg: float) -> folium.DivIcon:
    """
    单个三角箭头头（DivIcon），用 CSS rotate 旋转到 bearing 方向。
    """
    html = f"""
    <div style="
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-bottom: 16px solid {color};
        transform: rotate({bearing_deg:.2f}deg);
        transform-origin: 50% 60%;
        ">
    </div>
    """
    return folium.DivIcon(html=html)

def parse_fov_deg(exif: Dict[str, Any]) -> Optional[float]:
    # 你的 EXIF 里有 'FOV': '73.7 deg'
    return parse_float(exif.get("FOV"))

def parse_lrf_target(exif: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # 你的 EXIF 里 LRF 目标是 decimal（例如 22.3100105）
    tlat = parse_float(exif.get("LRFTargetLat"))
    tlon = parse_float(exif.get("LRFTargetLon"))
    tdist = parse_float(exif.get("LRFTargetDistance"))  # meters
    return tlat, tlon, tdist

# -------------------------
# Main
# -------------------------
def build_views_map(
    root_dir: str,
    out_html: str = "views_map.html",
    arrow_len_m: float = 3.0,     # 箭头长度（米）
    gap_m: float = 1.2,            # 线段末端留空（让箭头头后面“缺一段”更像方向指示）
    yaw_offset_deg: float = 0.0,   # 如果 yaw 方向整体偏转，可在这里统一加偏置
    pick_method: str = "first",
) -> str:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root_dir}")

    folders = [p for p in root.iterdir() if p.is_dir()]
    folders.sort(key=lambda p: p.name)

    if pick_method == "first":
        pick_func = pick_first_image
    elif pick_method == "last":
        pick_func = pick_last_image
    elif pick_method == "middle":
        pick_func = pick_middle_image
    else:
        raise ValueError(f"Unknown pick_method: {pick_method}")


    points: List[Dict[str, Any]] = []
    for i, folder in enumerate(folders, start=1):
        img = pick_func(folder)
        if img is None:
            print(f"[WARN] No image found in: {folder}")
            continue
        else:
            print(f"[INFO] Pick image: {img}")
        exif = get_exif(str(img))

        lat, lon = parse_gps_from_exif(exif)
        if lat is None or lon is None:
            print(f"[WARN] Missing GPS for {img}")
            continue

        # yaw：优先用云台 yaw（更接近相机视线方向）
        yaw = parse_float(exif.get("GimbalYawDegree"))
        if yaw is None:
            yaw = parse_float(exif.get("FlightYawDegree"))
        if yaw is None:
            yaw = 0.0
            print(f"[WARN] Missing yaw for {img}, fallback yaw=0.")

        yaw = (yaw + yaw_offset_deg) % 360.0

        fov_deg = parse_fov_deg(exif)
        tlat, tlon, tdist = parse_lrf_target(exif)

        points.append({
            "vid": f"V{i}",
            "folder": folder.name,
            "img": img.name,
            "lat": lat,
            "lon": lon,
            "yaw": yaw,
            "fov_deg": fov_deg,
            "tlat": tlat,
            "tlon": tlon,
            "tdist": tdist,
        })

    if not points:
        raise RuntimeError("No valid points collected. Check folders/images/EXIF parsing.")

    # center
    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)

    # ✅ 用 Esri 影像（一般可放大到 22+），并允许更大 max_zoom
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=21,
        max_zoom=24,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Clarity",
        name="Esri World Imagery (Clarity)",
        overlay=False,
        control=True,
        max_zoom=24,         # 允许地图放大的上限
        max_native_zoom=19,  # 底图服务器实际拥有的最高层级（通常为18或19）
    ).add_to(m)

    # 如果你也想让 OpenStreetMap 也能无限放大（虽然会糊，但不会白屏）
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        max_zoom=24,
        max_native_zoom=19,
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # draw arrows + labels
    for p in points:
        lat, lon = p["lat"], p["lon"]
        yaw = p["yaw"]

        # ✅ 方向修正：箭头方向 = 拍摄方向 yaw（不做 +180）
        arrow_bearing = yaw % 360.0
        color = bearing_to_color(arrow_bearing)

        # 线段末端留空 gap
        line_len = max(0.5, arrow_len_m - gap_m)
        lat_line_end, lon_line_end = forward_geodesic(lat, lon, arrow_bearing, line_len)
        lat_tip, lon_tip = forward_geodesic(lat, lon, arrow_bearing, arrow_len_m)

        # line
        folium.PolyLine(
            [(lat, lon), (lat_line_end, lon_line_end)],
            weight=4,
            opacity=0.9,
            color=color,
        ).add_to(m)

        # single triangle head
        folium.Marker(
            location=(lat_tip, lon_tip),
            icon=make_rotated_triangle_icon(color=color, bearing_deg=arrow_bearing),
        ).add_to(m)

        # 在 LRF 目标点处画一条“横向覆盖范围”红线
        tlat, tlon, tdist = p.get("tlat"), p.get("tlon"), p.get("tdist")
        fov_deg = p.get("fov_deg")

        if (tlat is not None) and (tlon is not None) and (tdist is not None) and (fov_deg is not None) and (tdist > 0) and (fov_deg > 0):
            half_width = tdist * math.tan(math.radians(fov_deg / 2.0))  # meters

            # 红线方向：与拍摄方向垂直（yaw ± 90）
            left_lat, left_lon = forward_geodesic(tlat, tlon, (arrow_bearing - 90.0) % 360.0, half_width)
            right_lat, right_lon = forward_geodesic(tlat, tlon, (arrow_bearing + 90.0) % 360.0, half_width)

            folium.PolyLine(
                [(left_lat, left_lon), (right_lat, right_lon)],
                weight=4,
                opacity=0.9,
                color="gray",
            ).add_to(m)
        # label
        folium.Marker(
            location=(lat, lon),
            tooltip=f'{p["vid"]} | {p["folder"]} | {p["img"]} | yaw={yaw:.2f}°',
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size: 12px;
                    font-weight: 700;
                    background: rgba(255,255,255,0.85);
                    padding: 2px 6px;
                    border-radius: 6px;
                    border: 1px solid rgba(0,0,0,0.25);
                    ">
                    {p["vid"]}
                </div>
            """),
        ).add_to(m)

    # fit bounds
    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    m.save(out_html)
    print(f"[OK] Saved map to: {out_html}")
    return out_html


if __name__ == "__main__":
    # TODO: 改成你的 35 个文件夹根目录（里面直接是 35 个子文件夹）
    ROOT_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\visible_views"
    build_views_map(
        ROOT_DIR,
        out_html="views_map.html",
        arrow_len_m=3.0,
        gap_m=1.2,
        yaw_offset_deg=0.0,  # 如果方向整体偏了（比如都顺时针/逆时针错 90°），改这里
    )

