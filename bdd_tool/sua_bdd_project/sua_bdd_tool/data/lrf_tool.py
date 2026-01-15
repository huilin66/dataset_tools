from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sua_bdd_tool.utils import load_json

def calculate_robust_wall_distance(img_files, exif_db, trim_ratio=0.05, bin_size=0.5, min_dist=1.0, max_dist=10.0):
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
        return 0.0

    # 在实际循环中读取
    raw_data = []
    for img_path in tqdm(sorted_files, desc="Reading Metadata"):
        # 确保img_path是Path对象
        path_obj = Path(img_path) if isinstance(img_path, str) else img_path
        meta = exif_db.get(path_obj.name, None)
        if meta is None:
            continue
        dist = meta['lrf_dist']
        if dist is None:
            continue
        # 过滤明显的错误数据 (比如 < min_dist 或 > max_dist)
        if min_dist < dist < max_dist:
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
    bins = np.arange(np.floor(min_val), np.ceil(max_val) + bin_size, bin_size)
    
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



def statistics_lrf_data(
    input_json_path, 
    output_repaired_json_path, 
    output_view_dist_json_path=None,
    min_dist=1.0,
    max_dist=10.0,
):
    """
    读取元数据 -> 按 View 分组 -> 计算鲁棒距离 -> 修复异常数据 -> 保存
    """
    print(f"正在读取元数据: {input_json_path} ...")
    exif_db = load_json(input_json_path)

    # --- 步骤 A: 按 rel_dir (View) 分组文件名 ---
    # 结构: { "flight1/view_01": ["img1.jpg", "img2.jpg"...], ... }
    view_groups = defaultdict(list)
    
    for filename, item in tqdm(exif_db.items(), desc="Search View"):
        rel_dir = item.get('rel_dir')
        if rel_dir:
            abs_path = Path(item['root_dir'])/Path(rel_dir)/Path(filename)
            view_groups[rel_dir].append(abs_path)

    print(f"共识别出 {len(view_groups)} 个 View，开始计算墙面距离...")

    # --- 步骤 B: 计算每个 View 的 Robust Wall Distance ---
    view_wall_distances = {}
    for view_name, img_files in tqdm(view_groups.items(), desc="Calculating Distances"):
        view_wall_distances[view_name] = calculate_robust_wall_distance(img_files, exif_db, min_dist=min_dist, max_dist=max_dist)

    # --- 步骤 C: 修复数据 ---
    repaired_count = 0
    unrepaired_count = 0
    for filename, item in exif_db.items():
        rel_dir = item.get('rel_dir')
        status = item.get('lrf_status')
        current_dist = item.get('lrf_dist')
        
        needs_repair = (status != 'Normal') or (current_dist is None or not (min_dist <= current_dist <= max_dist))
        if needs_repair:
            ref_dist = view_wall_distances[rel_dir]
            if ref_dist is not None:
                if current_dist is not None:
                    print(f"修复 {filename}：原始距离 {current_dist:.4f}m -> 修复为 {ref_dist:.4f}m")
                else:
                    print(f"修复 {filename}：原始距离 None -> 修复为 {ref_dist:.4f}m")
                item['lrf_status_original'] = item['lrf_status']
                item['lrf_lat_original'] = item['lrf_lat']
                item['lrf_lon_original'] = item['lrf_lon']
                item['lrf_dist_original'] = item['lrf_dist']

                item['lrf_status'] = item['lrf_status']+'_Repaired'
                item['lrf_lat'] = item['lat']
                item['lrf_lon'] = item['lon']
                item['lrf_dist'] = ref_dist
                
                repaired_count += 1
            else:
                unrepaired_count += 1


    # --- 步骤 D: 保存结果 ---
    
    # 1. 保存 View 距离表 (方便人工检查)
    if output_view_dist_json_path:
        Path(output_view_dist_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_view_dist_json_path, 'w', encoding='utf-8') as f:
            json.dump(view_wall_distances, f, indent=2, ensure_ascii=False)
            
    # 2. 保存修复后的完整 JSON
    Path(output_repaired_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_repaired_json_path, 'w', encoding='utf-8') as f:
        json.dump(exif_db, f, indent=2, ensure_ascii=False)

    print("\n处理完成！")
    print(f"View 距离表已保存至: {output_view_dist_json_path}")
    print(f"修复后的元数据已保存至: {output_repaired_json_path}")
    print(f"共修复异常记录: {repaired_count} 条")
