import os
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
import platform
import json


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

# ===================== 2. 核心处理流程 =====================

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

def yolo_dedup_pipeline(img_dir, yolo_txt_dir, output_dir, proj_params, 
                        iou_thresh, height_thresh_m, x_thresh_m=2.0,
                        target_classes=None, class_names_path=None, vis_font_size=24):
    
    # 路径准备
    dedup_label_dir = os.path.join(output_dir, "labels_dedup")
    vis_all_dir = os.path.join(output_dir, "vis_all")
    vis_by_id_dir = os.path.join(output_dir, "vis_by_id")
    group_info_path = os.path.join(output_dir, "labels_group_info.json")
    proj_info_path = os.path.join(output_dir, "project_info.json")
    
    os.makedirs(dedup_label_dir, exist_ok=True)
    os.makedirs(vis_all_dir, exist_ok=True)
    os.makedirs(vis_by_id_dir, exist_ok=True)

    # 0. 加载类别名称
    class_names = load_class_names(class_names_path)

    # 1. 读取并筛选
    all_dets = yolo_project2facade(img_dir, yolo_txt_dir, proj_params, target_classes)
    
    # 2. 去重
    all_dets_with_id = yolo_dedup(all_dets, iou_thresh, height_thresh_m)
    
    # 3. 按图片分组 (关键修复)
    dets_by_img = group_dets_by_image(all_dets_with_id)
    
    # 4. 写入
    yolo_dedup_write(dets_by_img, dedup_label_dir)
    
    # 5. 可视化
    dedup_vis(dets_by_img, img_dir, vis_all_dir, vis_by_id_dir, 
              class_names=class_names, font_size=vis_font_size)
    # 6. 生成调试用 JSON 报告
    export_debug_json(all_dets_with_id, group_info_path, class_names)

    # 7. 导出投影详情 JSON
    export_projection_details_json(all_dets_with_id, proj_info_path)

    print("🎉 完成：YOLO 投影去重 + ID 审计流水线")



if __name__ == "__main__":
    # ===================== 路径配置 =====================
    # 输入图片文件夹
    image_dir = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views\V30"
    # 输入 YOLO 标签文件夹
    yolo_dir = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\split_data\thermal_views_infer\V30\labels"
    # 输出根目录
    output_dir = r"e:\repository\dataset_tools\bdd_tool\sua_data_tools\yolo_dedup_out"
    
    # [新增] classes.txt 路径 (可选，如果没有则填 None)
    # 格式：每行一个类别名，第0行对应ID 0
    classes_txt_path = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\class.txt" 

    # ===================== 参数 =====================
    proj_params = {
        "focal_length_mm": 24.0,
        "sensor_height_mm": 24.0,
        "camera_height_m": 1.6,
        "facade_distance_m": 10.0,
    }
    
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

    yolo_dedup_pipeline(
        img_dir=image_dir, 
        yolo_txt_dir=yolo_dir, 
        output_dir=output_dir,
        proj_params=proj_params, 
        iou_thresh=iou_thresh, 
        height_thresh_m=height_thresh_m,
        target_classes=filter_classes,
        class_names_path=classes_txt_path,
        vis_font_size=vis_font_size,
        x_thresh_m=x_thresh_m,
    )