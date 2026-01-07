
# ======================
# 4. 主流程
# ======================

IMG_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views\V30"
YOLO_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views_infer\V30\labels"
OUT_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\yolo_dedup.geojson"
import os
import glob
import math
from tqdm import tqdm
from PIL import Image
import numpy as np

# ===================== 参数区 =====================

FOCAL_LENGTH_MM = 24.0      # 相机焦距（自己改）
SENSOR_HEIGHT_MM = 24.0     # 全画幅假设
CAMERA_HEIGHT_M = 1.6       # 相机离地高度（关键）
FACADE_DISTANCE_M = 10.0    # 相机到立面距离（近似即可）

IOU_THRESH = 0.5
HEIGHT_THRESH_M = 0.3       # 高度差阈值（立面核心参数）

os.makedirs(OUT_DIR, exist_ok=True)

# ===================== 工具函数 =====================
def parse_yolo_line(line):
    parts = list(map(float, line.strip().split()))
    if len(parts) == 5:
        cls, cx, cy, w, h = parts
        conf = 1.0
    elif len(parts) == 6:
        cls, cx, cy, w, h, conf = parts
    else:
        return None
    return int(cls), cx, cy, w, h, conf


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    px = cx * img_w
    py = cy * img_h
    bw = w * img_w
    bh = h * img_h
    return px, py, bw, bh


def project_to_facade(px, py, img_w, img_h):
    """
    像素 → 立面 X,Z（米）
    """
    fy = img_h * (FOCAL_LENGTH_MM / SENSOR_HEIGHT_MM)

    z = CAMERA_HEIGHT_M + (img_h / 2 - py) / fy * FACADE_DISTANCE_M
    x = (px - img_w / 2) / fy * FACADE_DISTANCE_M
    return x, z


def iou_1d(a_min, a_max, b_min, b_max):
    inter = max(0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    return inter / union if union > 0 else 0


# ===================== 主流程 =====================
all_dets = []
global_id = 0

print("📥 读取 YOLO + 投影到立面坐标...")

for txt_path in tqdm(glob.glob(os.path.join(YOLO_DIR, "*.txt"))):
    name = os.path.splitext(os.path.basename(txt_path))[0]
    img_path = os.path.join(IMG_DIR, name + ".jpg")
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    img_w, img_h = img.size

    with open(txt_path, "r") as f:
        for line in f:
            parsed = parse_yolo_line(line)
            if parsed is None:
                continue

            cls, cx, cy, w, h, conf = parsed
            px, py, bw, bh = yolo_to_pixel(cx, cy, w, h, img_w, img_h)
            x, z = project_to_facade(px, py, img_w, img_h)

            det = {
                "gid": global_id,
                "img": name,
                "cls": cls,
                "conf": conf,
                "x": x,
                "z": z,
                "h": bh / img_h * FACADE_DISTANCE_M,
                "raw": (cls, cx, cy, w, h, conf),
            }
            all_dets.append(det)
            global_id += 1

print(f"✅ 共投影 {len(all_dets)} 个 YOLO 检测")

# ===================== 去重 =====================
print("🧹 基于立面高度去重...")

keep = []
used = set()

for i, a in enumerate(tqdm(all_dets)):
    if a["gid"] in used:
        continue

    for b in all_dets[i + 1:]:
        if b["gid"] in used:
            continue
        if a["cls"] != b["cls"]:
            continue

        # 高度 IoU
        a_min, a_max = a["z"] - a["h"]/2, a["z"] + a["h"]/2
        b_min, b_max = b["z"] - b["h"]/2, b["z"] + b["h"]/2

        if abs(a["z"] - b["z"]) < HEIGHT_THRESH_M and \
           iou_1d(a_min, a_max, b_min, b_max) > IOU_THRESH:

            # 保留置信度高的
            if a["conf"] >= b["conf"]:
                used.add(b["gid"])
            else:
                used.add(a["gid"])
                break

    if a["gid"] not in used:
        keep.append(a)

print(f"✅ 去重后剩余 {len(keep)} 个")

# ===================== 写回 YOLO =====================
print("✍️ 写回 YOLO txt...")

out_map = {}
for d in keep:
    out_map.setdefault(d["img"], []).append(d)

for img, dets in out_map.items():
    out_path = os.path.join(OUT_DIR, img + ".txt")
    with open(out_path, "w") as f:
        for d in dets:
            cls, cx, cy, w, h, conf = d["raw"]
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

print("🎉 完成：YOLO 立面投影去重")
