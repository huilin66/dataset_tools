import time
import os, re, math, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tqdm import tqdm

# -----------------------------
# 你已有的 EXIF 读取函数
# -----------------------------
def pyexif_to_dict(img_path):
    import pyexif
    img = pyexif.ExifEditor(str(img_path))
    return img.getDictTags()


# -----------------------------
# 工具：健壮解析
# -----------------------------
def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    s = re.sub(r"[^0-9eE\.\-\+]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def _parse_exif_rational(x: Any) -> Optional[float]:
    """
    Parse EXIF rational formats:
    - "1312/100" -> 13.12
    - "29/1 33/1 1312/100" -> None (handled elsewhere)
    - numeric -> float
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def parse_gps_coord(value: Any) -> Optional[float]:
    """
    Parse one coordinate (lat or lon) into decimal degrees.
    Supports:
    1) float/int already in degrees
    2) DMS strings like:
       - "29 deg 33' 13.12\""
       - "29°33'13.12\""
       - "29 33 13.12"
    3) Tuple/list like:
       - [29, 33, 13.12]
       - ["29/1","33/1","1312/100"]
    4) Space-separated rationals:
       - "29/1 33/1 1312/100"
    """
    if value is None:
        return None

    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)

    # If it's list/tuple of DMS parts
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        parts = list(value)
        deg = _parse_exif_rational(parts[0])
        minute = _parse_exif_rational(parts[1]) if len(parts) >= 2 else 0.0
        sec = _parse_exif_rational(parts[2]) if len(parts) >= 3 else 0.0
        if deg is None or minute is None or sec is None:
            return None
        return float(deg) + float(minute) / 60.0 + float(sec) / 3600.0

    s = str(value).strip()

    # Space-separated rationals: "29/1 33/1 1312/100"
    if re.search(r"\d+/\d+", s) and " " in s:
        toks = s.split()
        vals = [_parse_exif_rational(t) for t in toks[:3]]
        if any(v is None for v in vals[:2]):
            return None
        deg, minute = vals[0], vals[1]
        sec = vals[2] if len(vals) >= 3 and vals[2] is not None else 0.0
        return float(deg) + float(minute) / 60.0 + float(sec) / 3600.0

    # DMS string with deg/min/sec
    # examples: 29 deg 33' 13.12", 29°33'13.12", 29 33 13.12
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if len(m) >= 3:
        deg, minute, sec = map(float, m[:3])
        return deg + minute / 60.0 + sec / 3600.0
    elif len(m) == 2:
        deg, minute = map(float, m[:2])
        return deg + minute / 60.0
    elif len(m) == 1:
        # could already be decimal degrees as string
        try:
            return float(m[0])
        except Exception:
            return None

    return None


def parse_latlon_from_exif(ex: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Robust lat/lon parsing with Ref sign.
    Tries:
    - GPSLatitude/GPSLongitude (+ GPSLatitudeRef/GPSLongitudeRef)
    - GPSPosition "lat lon" / "lat, lon"
    """
    # 1) Standard split fields
    lat_raw = ex.get("GPSLatitude")
    lon_raw = ex.get("GPSLongitude")
    if lat_raw is not None and lon_raw is not None:
        lat = parse_gps_coord(lat_raw)
        lon = parse_gps_coord(lon_raw)

        # apply hemisphere refs
        lat_ref = str(ex.get("GPSLatitudeRef", "")).strip().upper()
        lon_ref = str(ex.get("GPSLongitudeRef", "")).strip().upper()
        if lat is not None and lat_ref in ("S", "SOUTH"):
            lat = -abs(lat)
        if lon is not None and lon_ref in ("W", "WEST"):
            lon = -abs(lon)

        if lat is not None and lon is not None:
            return lat, lon

    # 2) DJI combined field
    gps_pos = ex.get("GPSPosition") or ex.get("GPS Position") or ex.get("Composite:GPSPosition") or ex.get("XMP:GPSPosition")
    if gps_pos is not None:
        s = str(gps_pos).strip()
        parts = re.split(r"[,\s]+", s)
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            lat = parse_gps_coord(parts[0])

def _to_int(x) -> Optional[int]:
    f = _to_float(x)
    return None if f is None else int(round(f))

def pick(ex: Dict[str, Any], *keys: str):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return None

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def build_exif_cache(images_dir: str, img_files: List[str], cache_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Cache EXIF per image to avoid re-reading every run.

    Cache format:
    {
      "__meta__": {"images_dir": "...", "created": "..."},
      "DJI_xxx.jpg": {"k1": v1, ...},
      ...
    }

    Update strategy:
    - If image not in cache -> read and add
    - If image mtime changed -> re-read and update
    """
    cache_file = Path(cache_path)
    cache: Dict[str, Any] = {}

    if cache_file.exists():
        try:
            cache = json.load(open(cache_file, "r", encoding="utf-8"))
        except Exception:
            cache = {}

    meta = cache.get("__meta__", {})
    data = {k: v for k, v in cache.items() if k != "__meta__"}

    # store per-image mtime in cache to detect changes
    mtime_key = "__mtime__"

    updated = 0
    added = 0
    t0 = time.time()

    for fn in tqdm(img_files, desc="Loading EXIF (cached)"):
        img_path = Path(images_dir) / fn
        mtime = img_path.stat().st_mtime

        need_read = False
        if fn not in data:
            need_read = True
        else:
            prev_mtime = data[fn].get(mtime_key, None)
            if prev_mtime is None or abs(prev_mtime - mtime) > 1e-6:
                need_read = True

        if need_read:
            ex = pyexif_to_dict(img_path)
            ex[mtime_key] = mtime
            data[fn] = ex
            if fn in cache:
                updated += 1
            else:
                added += 1

    # write back
    out_cache = {"__meta__": {"images_dir": str(images_dir), "updated_at": time.time()}}
    out_cache.update(data)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_cache, open(cache_file, "w", encoding="utf-8"), ensure_ascii=False)

    t1 = time.time()
    print(f"[INFO] EXIF cache: {len(data)} items | added={added} updated={updated} | took {t1-t0:.1f}s")
    return data

# -----------------------------
# YOLO labels: cls cx cy w h (norm) [conf]
# -----------------------------
def parse_yolo_txt(label_path: str) -> List[dict]:
    dets = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(float(p[0]))
            cx, cy, w, h = map(float, p[1:5])
            conf = float(p[5]) if len(p) >= 6 else None
            dets.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "conf": conf})
    return dets


# -----------------------------
# WGS84 -> ENU (meters)
# -----------------------------
@dataclass
class ENUFrame:
    lat0: float
    lon0: float
    alt0: float
    to_ecef: Transformer
    R_ecef2enu: np.ndarray
    x0: float
    y0: float
    z0: float
# def parse_latlon_from_exif(ex: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
#     """
#     Try multiple DJI/EXIF key patterns:
#     - GPSLatitude/GPSLongitude
#     - GPSPosition: "lat lon" or "lat, lon"
#     - variations: "GPS Position", "Composite:GPSPosition", etc.
#     """
#     # 1) Standard EXIF split fields
#     lat = _to_float(ex.get("GPSLatitude"))
#     lon = _to_float(ex.get("GPSLongitude"))
#     if lat is not None and lon is not None:
#         return lat, lon

#     # 2) DJI common combined field
#     gps_pos = pick(ex, "GPSPosition", "GPS Position", "Composite:GPSPosition", "XMP:GPSPosition")
#     if gps_pos is not None:
#         s = str(gps_pos).strip()
#         # allow "lat lon" / "lat,lon" / "lat, lon"
#         parts = re.split(r"[,\s]+", s)
#         parts = [p for p in parts if p != ""]
#         if len(parts) >= 2:
#             lat2 = _to_float(parts[0])
#             lon2 = _to_float(parts[1])
#             if lat2 is not None and lon2 is not None:
#                 return lat2, lon2

#     # 3) Sometimes stored under different names
#     lat = _to_float(pick(ex, "Latitude", "lat", "GPSLat"))
#     lon = _to_float(pick(ex, "Longitude", "lon", "GPSLon"))
#     if lat is not None and lon is not None:
#         return lat, lon

#     return None, None

def build_enu_frame(exif_list: List[Dict[str, Any]]) -> ENUFrame:
    first = None
    lat0 = lon0 = None

    for ex in exif_list:
        lat, lon = parse_latlon_from_exif(ex)
        if lat is not None and lon is not None:
            first = ex
            lat0, lon0 = lat, lon
            break

    if first is None:
        # 额外打印一下第一个样本有哪些 key，帮助你确认字段名
        sample_keys = sorted(list(exif_list[0].keys())) if exif_list else []
        raise RuntimeError(
            "No GPS found. Tried GPSLatitude/GPSLongitude and GPSPosition.\n"
            f"Sample keys (first image): {sample_keys[:60]} ... (total {len(sample_keys)})"
        )

    alt0 = float(_to_float(pick(first, "AbsoluteAltitude", "GPSAltitude")) or 0.0)

    to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    x0, y0, z0 = to_ecef.transform(lon0, lat0, alt0)

    lam = math.radians(lon0)
    phi = math.radians(lat0)
    R = np.array([
        [-math.sin(lam),                 math.cos(lam),                0.0],
        [-math.sin(phi)*math.cos(lam),  -math.sin(phi)*math.sin(lam),  math.cos(phi)],
        [ math.cos(phi)*math.cos(lam),   math.cos(phi)*math.sin(lam),  math.sin(phi)]
    ], dtype=np.float64)

    return ENUFrame(lat0, lon0, alt0, to_ecef, R, x0, y0, z0)


def wgs84_to_enu(frame: ENUFrame, lat: float, lon: float, alt: float) -> np.ndarray:
    x, y, z = frame.to_ecef.transform(lon, lat, alt)
    v = np.array([x - frame.x0, y - frame.y0, z - frame.z0], dtype=np.float64)
    return frame.R_ecef2enu @ v


# -----------------------------
# 平面拟合（LRF 点 -> 多面墙）
# -----------------------------
@dataclass
class Plane:
    plane_id: int
    n: np.ndarray
    p0: np.ndarray
    inliers_idx: List[int]

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p0 = points.mean(axis=0)
    X = points - p0
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, p0

def ransac_multi_planes(
    pts: np.ndarray,
    dist_thresh: float = 0.20,
    min_inliers: int = 25,
    max_planes: int = 20,
    iters: int = 4000,
    random_state: int = 0
) -> List[Plane]:
    if len(pts) < min_inliers:
        return []

    rng = np.random.RandomState(random_state)
    remaining = np.arange(len(pts))
    planes: List[Plane] = []
    pid = 1

    def fit_plane_from_3(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None, None
        n = n / nn
        p0 = (p1 + p2 + p3) / 3.0
        return n, p0

    while len(remaining) >= min_inliers and len(planes) < max_planes:
        P = pts[remaining]
        best_inliers = None
        best_count = 0
        best_n = None
        best_p0 = None

        for _ in range(iters):
            i, j, k = rng.choice(len(P), size=3, replace=False)
            n, p0 = fit_plane_from_3(P[i], P[j], P[k])
            if n is None:
                continue
            d = np.abs((P - p0) @ n)
            inliers = np.where(d <= dist_thresh)[0]
            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers
                best_n, best_p0 = n, p0

        if best_inliers is None or best_count < min_inliers:
            break

        inlier_global = remaining[best_inliers]
        n_ref, p0_ref = fit_plane_svd(pts[inlier_global])
        planes.append(Plane(pid, n_ref, p0_ref, inlier_global.tolist()))
        pid += 1

        keep = np.ones(len(remaining), dtype=bool)
        keep[best_inliers] = False
        remaining = remaining[keep]

    return planes


def merge_similar_planes(planes: List[Plane], angle_deg: float = 12.0, offset_m: float = 0.40) -> List[Plane]:
    if not planes:
        return []

    def ang(n1, n2):
        c = np.clip(np.abs(float(n1 @ n2)), -1.0, 1.0)
        return math.degrees(math.acos(c))

    merged: List[Plane] = []
    used = [False] * len(planes)

    for i, pi in enumerate(planes):
        if used[i]:
            continue
        group = [pi]
        used[i] = True
        for j, pj in enumerate(planes):
            if used[j]:
                continue
            if ang(pi.n, pj.n) <= angle_deg:
                off = abs(float(pi.n @ (pj.p0 - pi.p0)))
                if off <= offset_m:
                    group.append(pj)
                    used[j] = True
        all_idx = sorted({k for g in group for k in g.inliers_idx})
        merged.append(Plane(len(merged)+1, pi.n, pi.p0, all_idx))
    return merged


# -----------------------------
# 投影：像素 -> 射线 -> 平面交点（ENU, meters）
# -----------------------------
def rpy_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx

def build_K_approx(img_w: int, img_h: int) -> np.ndarray:
    cx, cy = img_w / 2.0, img_h / 2.0
    fov_deg = 70.0
    fx = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy = fx
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def pixel_ray_world(u: float, v: float, K: np.ndarray, R_wc: np.ndarray) -> np.ndarray:
    x_c = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
    x_c = x_c / (np.linalg.norm(x_c) + 1e-12)
    d_w = R_wc @ x_c
    return d_w / (np.linalg.norm(d_w) + 1e-12)

def intersect_ray_plane(C: np.ndarray, d: np.ndarray, n: np.ndarray, p0: np.ndarray) -> Optional[np.ndarray]:
    denom = float(n @ d)
    if abs(denom) < 1e-8:
        return None
    t = float(n @ (p0 - C)) / denom
    if t <= 0:
        return None
    return C + t * d


# -----------------------------
# 主流程：输出 ENU 米坐标 + object_id
# -----------------------------
def run_dedup_enu(
    images_dir: str,
    labels_dir: str,
    out_json_path: str,
    # 平面参数
    plane_dist_thresh_m: float = 0.20,
    plane_min_inliers: int = 25,
    plane_max_planes: int = 12,
    plane_merge_angle_deg: float = 12.0,
    plane_merge_offset_m: float = 0.40,
    # 去重聚类参数（对 3D 点聚类）
    dedup_eps_m: float = 0.30,
    dedup_min_samples: int = 1,
    conf_thresh: Optional[float] = None,
    # 姿态修正：如果投影失败多，先试把它设为 True
    flip_pitch: bool = False,
):
    images_dir = str(images_dir)
    labels_dir = str(labels_dir)

    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                       key=natural_key)
    print(f"[INFO] images: {len(img_files)}")

    # 1) 读取 EXIF（pyexif）
    cache_path = os.path.join(images_dir, "_exif_cache.json")
    exif_map = build_exif_cache(images_dir, img_files, cache_path)
    exif_list = list(exif_map.values())

    frame = build_enu_frame(exif_list)

    # 2) LRF 点 -> ENU
    lrf_pts = []
    lrf_img = []
    for fn in tqdm(img_files, desc="Collecting LRF points"):
        ex = exif_map[fn]
        tlat = _to_float(pick(ex, "LRFTargetLat"))
        tlon = _to_float(pick(ex, "LRFTargetLon"))
        talt = _to_float(pick(ex, "LRFTargetAlt", "LRFTargetAbsAlt"))
        if tlat is None or tlon is None or talt is None:
            continue
        lrf_pts.append(wgs84_to_enu(frame, tlat, tlon, talt))
        lrf_img.append(fn)

    lrf_pts = np.array(lrf_pts, dtype=np.float64)
    print(f"[INFO] LRF points: {len(lrf_pts)} (need >= {plane_min_inliers})")
    if len(lrf_pts) < plane_min_inliers:
        raise RuntimeError("Too few LRF points. Ensure LRFTargetLat/Lon/Alt are available.")

    # 3) 自动多平面
    planes = ransac_multi_planes(
        lrf_pts,
        dist_thresh=plane_dist_thresh_m,
        min_inliers=plane_min_inliers,
        max_planes=plane_max_planes
    )
    planes = merge_similar_planes(planes, angle_deg=plane_merge_angle_deg, offset_m=plane_merge_offset_m)
    if not planes:
        raise RuntimeError("No planes found. Try increasing plane_dist_thresh_m or lowering plane_min_inliers.")

    # refine planes
    refined = []
    for i, pl in enumerate(planes, start=1):
        pts = lrf_pts[np.array(pl.inliers_idx, dtype=int)]
        n, p0 = fit_plane_svd(pts)
        refined.append(Plane(i, n, p0, pl.inliers_idx))
    planes = refined
    print(f"[INFO] planes found: {len(planes)}")

    # 4) 每张图分配平面（用它自己的 LRF 点 -> 最近平面）
    plane_for_image: Dict[str, int] = {}
    for fn in tqdm(img_files, desc="Assigning plane per image"):
        ex = exif_map[fn]
        tlat = _to_float(pick(ex, "LRFTargetLat"))
        tlon = _to_float(pick(ex, "LRFTargetLon"))
        talt = _to_float(pick(ex, "LRFTargetAlt", "LRFTargetAbsAlt"))
        if tlat is None or tlon is None or talt is None:
            continue
        T = wgs84_to_enu(frame, tlat, tlon, talt)
        best = None
        for pl in planes:
            dist = abs(float(pl.n @ (T - pl.p0)))
            if best is None or dist < best[0]:
                best = (dist, pl.plane_id)
        plane_for_image[fn] = best[1]
    print(f"[INFO] images with plane assignment: {len(plane_for_image)}")

    plane_map = {p.plane_id: p for p in planes}

    # 5) 投影 YOLO 检测 -> 3D 点（ENU meters）
    det_rows = []
    n_total_labels = 0
    n_project_ok = 0
    n_missing_label = 0

    for fn in tqdm(img_files, desc="Projecting detections to ENU"):
        stem = os.path.splitext(fn)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            n_missing_label += 1
            continue

        dets = parse_yolo_txt(label_path)
        if conf_thresh is not None:
            dets = [d for d in dets if (d["conf"] is None or d["conf"] >= conf_thresh)]
        if not dets:
            continue

        n_total_labels += len(dets)

        if fn not in plane_for_image:
            continue
        pid = plane_for_image[fn]
        pl = plane_map[pid]

        ex = exif_map[fn]

        lat, lon = parse_latlon_from_exif(ex)
        alt = _to_float(pick(ex, "AbsoluteAltitude", "GPSAltitude")) or 0.0
        if lat is None or lon is None:
            continue
        C = wgs84_to_enu(frame, lat, lon, alt)

        yaw = _to_float(pick(ex, "GimbalYawDegree", "FlightYawDegree")) or 0.0
        pitch = _to_float(pick(ex, "GimbalPitchDegree", "FlightPitchDegree")) or 0.0
        roll = _to_float(pick(ex, "GimbalRollDegree", "FlightRollDegree")) or 0.0
        if flip_pitch:
            pitch = -pitch

        img_w = _to_int(pick(ex, "ImageWidth"))
        img_h = _to_int(pick(ex, "ImageHeight"))
        if img_w is None or img_h is None:
            continue

        K = build_K_approx(img_w, img_h)
        R_wc = rpy_to_R(yaw, pitch, roll)

        for di, d in enumerate(dets):
            px = d["cx"] * img_w
            py = d["cy"] * img_h
            dw = pixel_ray_world(px, py, K, R_wc)
            P = intersect_ray_plane(C, dw, pl.n, pl.p0)
            if P is None:
                continue
            n_project_ok += 1
            det_rows.append({
                "image": fn,
                "det_idx": di,
                "cls": int(d["cls"]),
                "conf": None if d["conf"] is None else float(d["conf"]),
                "plane_id": int(pid),
                "X": float(P[0]), "Y": float(P[1]), "Z": float(P[2]),  # ENU meters
            })

    print(f"[INFO] missing label files: {n_missing_label}")
    print(f"[INFO] detections in labels: {n_total_labels}, projected OK: {n_project_ok}")
    if n_project_ok == 0:
        raise RuntimeError("No projected points. Try flip_pitch=True first.")

    # 6) 去重：对 (plane_id, cls) 内的 3D 点做 DBSCAN -> object_id
    next_id = 1
    for pid in sorted(set(r["plane_id"] for r in det_rows)):
        idx_plane = [i for i, r in enumerate(det_rows) if r["plane_id"] == pid]
        for cls in sorted(set(det_rows[i]["cls"] for i in idx_plane)):
            idxs = [i for i in idx_plane if det_rows[i]["cls"] == cls]
            X = np.array([[det_rows[i]["X"], det_rows[i]["Y"], det_rows[i]["Z"]] for i in idxs], dtype=np.float64)
            if len(X) == 1:
                det_rows[idxs[0]]["object_id"] = next_id
                next_id += 1
                continue
            labels = DBSCAN(eps=dedup_eps_m, min_samples=dedup_min_samples).fit_predict(X)
            for lb in set(labels):
                members = [idxs[k] for k in range(len(idxs)) if labels[k] == lb]
                if lb == -1:
                    for m in members:
                        det_rows[m]["object_id"] = next_id
                        next_id += 1
                else:
                    oid = next_id
                    next_id += 1
                    for m in members:
                        det_rows[m]["object_id"] = oid

    out = {
        "coord_sys": "ENU meters (origin = first GPS in dataset)",
        "planes": [{"plane_id": p.plane_id, "n": p.n.tolist(), "p0": p.p0.tolist(),
                    "num_lrf_inliers": len(p.inliers_idx)} for p in planes],
        "params": {
            "flip_pitch": flip_pitch,
            "plane_dist_thresh_m": plane_dist_thresh_m,
            "plane_min_inliers": plane_min_inliers,
            "plane_max_planes": plane_max_planes,
            "plane_merge_angle_deg": plane_merge_angle_deg,
            "plane_merge_offset_m": plane_merge_offset_m,
            "dedup_eps_m": dedup_eps_m,
            "dedup_min_samples": dedup_min_samples,
            "conf_thresh": conf_thresh,
        },
        "detections": det_rows,
        "unique_object_ids": len(set(r["object_id"] for r in det_rows)),
    }

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] unique_object_ids: {out['unique_object_ids']}")
    print(f"[OK] wrote: {out_json_path}")


if __name__ == "__main__":
    images_dir = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\thermal"
    labels_dir = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\thermal_infer_without_conf\labels"
    out_json_path = r"./dedup_enu_object_id.json"

    # 第一次建议先 flip_pitch=False 跑一遍；
    # 如果 projected OK 很低或为 0，就改 flip_pitch=True 再跑。
    run_dedup_enu(
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_json_path=out_json_path,
        plane_dist_thresh_m=0.20,
        plane_min_inliers=25,
        plane_max_planes=25,
        plane_merge_angle_deg=12.0,
        plane_merge_offset_m=0.40,
        dedup_eps_m=0.30,
        dedup_min_samples=1,
        conf_thresh=None,
        flip_pitch=True
    )
