import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from folium.utilities import none_max
from tqdm import tqdm

import config
from sua_bdd_tool.utils import yolo_write
from sua_bdd_tool.utils.file_opt import find_all_images
from sua_bdd_tool.utils.projection import compute_iou_2d, iou_1d
from sua_bdd_tool.utils.visualization import get_class_color, get_contrasting_text_color


def project_adaptive(px, py, W, H, meta, wall_distance_m):
    """
    自适应投影函数
    :param meta: 当前图片的元数据 (alt, pitch, focal_35mm)
    :param wall_distance_m: 全局固定的墙面距离
    """

    # 0. 【核心修正】将 LRF 直线距离转换为水平距离
    # 假设 lrf_distance_m 是激光测距得到的直线距离
    # 假设 pitch 水平为0，向上为正，向下为负
    pitch_rad = math.radians(meta['pitch'])

    # 水平距离 = 直线距离 * cos(云台俯仰角)
    # 注意：这里假设激光打在图片中心点 (cx, cy)。如果激光没对准中心，会有误差，但通常可忽略。
    dist_horizontal = wall_distance_m * math.cos(pitch_rad)

    # 1. 计算像素焦距 (基于 35mm 等效焦距)
    # 35mm 全画幅传感器宽度为 36mm
    # fx_pix = (F_35mm / 36mm) * ImageWidth_pix
    fx = (meta['focal_35'] / 36.0) * W
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
    theta_total = pitch_rad + alpha_y
    
    # 4. 计算物理坐标
    # Z (高度) = 无人机高度 + 垂直增量
    # 垂直增量 = 距离 * tan(总角度)
    # z = meta['abs_alt'] + wall_distance_m * math.tan(theta_total)
    z = meta['abs_alt'] + dist_horizontal * math.tan(theta_total)

    # X (水平) = 距离 * tan(水平夹角) / cos(垂直夹角修正)
    # 简单近似：x = u / fx * distance
    x = (u / fx) * wall_distance_m
    
    # H (物体高度) = 像素高 / fy * 距离
    # 严格来说也受 pitch 影响，但物体很小时可忽略
    h_obj = (py / H) * 0 # 这是一个占位，实际应该用 box_h
    
    return x, z

# ===================== 2. 核心处理流程 =====================
def yolo_projecting(img_dir, yolo_txt_dir, exif_db, floor_manager, global_wall_dist, target_classes=None):
    all_dets = []
    gid = 0

    # 1. 获取所有图片列表
    img_files = find_all_images(img_dir)
    
    if not img_files:
        print("❌ 未找到图片")
        return None

    print("📥 读取 YOLO + 自适应投影 (Metadata Driven)...")

    for img_path in tqdm(img_files):
        image_name = Path(img_path).name
        stem = Path(img_path).stem
        txt_path = os.path.join(yolo_txt_dir, stem + ".txt")
        
        if not os.path.exists(txt_path):
            continue

        # 读取图片尺寸
        with Image.open(img_path) as img:
            W, H = img.size
            
        # 【关键步骤】读取当前图片的元数据
        meta = exif_db.get(image_name, none_max)

        with open(txt_path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) < 5: continue
                
                cls = int(vals[0])
                if target_classes and cls not in target_classes: continue
                
                cx, cy, w, h = vals[1:5]
                conf = vals[5] if len(vals) > 5 else 1.0
                
                # 1. 还原像素坐标 (Pixel Coordinates)
                # 计算左上角 (x1, y1) 和 右下角 (x2, y2)
                # YOLO 格式是 center_x, center_y, width, height
                px, py, bw, bh = cx*W, cy*H, w*W, h*H

                # # 1. 投影计算 (得到绝对海拔 Z)
                # world_x, world_z_abs = project_adaptive(px, py, W, H, meta, global_wall_dist)

                x1_pix, y1_pix, x2_pix, y2_pix = px - bw/2, py - bh/2, px + bw/2, py + bh/2

                # 2. 【关键】分别投影 Top-Left 和 Bottom-Right
                # 投影左上角 -> 得到墙面上物体的 左边界(Lx) 和 上边界(Tz)
                world_x_left, world_z_top = project_adaptive(x1_pix, y1_pix, W, H, meta, meta['lrf_dist'])
                
                # 投影右下角 -> 得到墙面上物体的 右边界(Rx) 和 下边界(Bz)
                world_x_right, world_z_bottom = project_adaptive(x2_pix, y2_pix, W, H, meta, meta['lrf_dist'])
                
                # 3. 重构物理属性
                # 物理中心 X
                world_x = (world_x_left + world_x_right) / 2
                
                # 物理中心 Z
                world_z_abs = (world_z_top + world_z_bottom) / 2
                
                # 物理宽度 (由于透视，左右边界的投影距离可能不完全对称，取差值绝对值)
                real_w = abs(world_x_right - world_x_left)
                
                # 物理高度 (这是解决“虚高”的关键)
                real_h = abs(world_z_top - world_z_bottom)
              
                
                # 2. [新增] 楼层计算
                # 直接传入绝对海拔 Z
                floor_name = floor_manager.get_floor(world_z_abs)
                
                # 计算物体实际高度
                # fx = (meta['focal_35'] / 36.0) * W
                # real_h = (bh / fx) * global_wall_dist
                all_dets.append({
                    "gid": gid,
                    "img": image_name,
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
                    "meta_alt": meta['abs_alt'],
                    "meta_lrf": meta['lrf_dist'] 
                })
                gid += 1
    print(f"loaded {len(all_dets)} boxes from {len(img_files)} images")
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


def yolo_grouping(all_dets, iou_thresh=0.5, height_thresh_m=0.3, x_thresh_m=1.5):
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

            # ===================== 新增修复逻辑 ====================
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
    for d in tqdm(all_dets, desc="grouping detections by image"):
        by_img.setdefault(d["img"], []).append(d)
    print(f"📄 grouping images finish, total {len(by_img)} images")
    return by_img


def dets_write(dets_by_img, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img_name, dets in tqdm(dets_by_img.items(), desc="writing detections to files"):
        yolo_write(dets, os.path.join(save_dir, Path(img_name).with_suffix('.txt')), result_format='dict')
    print(f"✅ 写入 {len(dets_by_img)} 个文件到 {save_dir}")


def analyze_and_vis_conflicts(dets_by_img, img_dir, output_dir, class_names=None, vis_font_size=20, vis=True):
    if not vis:
        return
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
                        cls_color = get_class_color(d['cls'], config.COLOR_PALETTE)

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


def export_grouping_info(all_dets, output_path, class_names=None):
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

