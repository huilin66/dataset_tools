from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from direction_check import build_views_map

RGB_SUFFIX = "_V.JPG"
T_SUFFIX = "_T.JPG"


# region rgb-t index

@dataclass
class ImageItem:
    id: str
    flight_folder: str
    rgb_dir: str
    rgb_name: str
    t_dir: Optional[str] = None
    t_name: Optional[str] = None

def is_rgb(name: str) -> bool:
    return name.upper().endswith(RGB_SUFFIX)

def rgb_to_t_name(rgb_name: str) -> str:
    # DJI_..._V.JPG -> DJI_..._T.JPG
    return rgb_name[:-len(RGB_SUFFIX)] + T_SUFFIX

def build_index(data_path, index_path=None) -> List[ImageItem]:
    """
    改进版：基于序列号(Sequence Number)配对，解决时间戳秒数偏差问题。
    """
    data_root = Path(data_path)
    items: List[ImageItem] = []
    idx = 0

    # 正则表达式：匹配 DJI 文件名，捕获序列号和类型(V或T)
    # 兼容格式：DJI_2025..._0003_V.JPG
    pattern = re.compile(r'DJI_.*_(\d{4})_([VT])\.JPG$')

    for folder in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        flight_folder = folder.name
        
        # 1. 预扫描当前文件夹：按序列号归类
        seq_map = defaultdict(dict)
        all_files = list(folder.iterdir())
        
        for p in all_files:
            if not p.is_file():
                continue
            
            match = pattern.search(p.name)
            if match:
                seq_num = match.group(1)  # 序列号，如 0003
                img_type = match.group(2) # 类型，V 或 T
                seq_map[seq_num][img_type] = p
        
        # 2. 遍历序列号映射表，构建 ImageItem
        # 按序列号排序，确保处理顺序
        sorted_seqs = sorted(seq_map.keys())
        for seq_num in tqdm(sorted_seqs, desc=f"Scan {flight_folder} to build index"):
            pair = seq_map[seq_num]
            
            # 以 V（可见光）为基准
            if 'V' in pair:
                rgb_path = pair['V']
                t_path = pair.get('T') # 如果没有 T，则为 None
                
                items.append(
                    ImageItem(
                        id=f"{flight_folder}:{idx:06d}",
                        flight_folder=flight_folder,
                        rgb_dir=str(rgb_path.parent),
                        rgb_name=rgb_path.name,
                        t_dir=str(t_path.parent) if t_path else None,
                        t_name=t_path.name if t_path else None,
                    )
                )
                idx += 1
    
    print('Scan Info:')
    print(f'Total RGB images: {len(items)}')
    print(f'Successfully paired (RGBT): {sum(1 for x in items if x.t_name)} pairs')

    save_index(items, Path(index_path))
    return items

def save_index(items: List[ImageItem], out_path: Path) -> None:
    out = [asdict(x) for x in items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'save {out_path}')

def load_index(index_path) -> List[ImageItem]:
    """
    从JSON文件加载图像索引数据。

    Args:
        path (Path): JSON文件路径，包含图像索引数据

    Returns:
        List[ImageItem]: 图像项目列表，每个项目包含图像的元数据信息
    """
    if isinstance(index_path, str):
        index_path = Path(index_path)
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return [ImageItem(**x) for x in data]

# endregion


# region copy files

def export_dataset(index_path: str, output_root: str):
    """
    将所有数据拷贝到指定文件夹下的 visible 和 thermal 目录
    """
    items = load_index(Path(index_path))

    # 1. 准备目标路径
    out_path = Path(output_root)
    vis_dir = out_path / "visible"
    thr_dir = out_path / "thermal"

    # 递归创建目录（如果不存在）
    vis_dir.mkdir(parents=True, exist_ok=True)
    thr_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始拷贝数据到: {out_path} ...")

    for item in tqdm(items, desc="Copying images"):
        # --- 处理可见光照片 ---
        src_rgb = Path(item.rgb_dir) / item.rgb_name
        if src_rgb.exists():
            # 为了防止重名，目标文件名加上 flight_folder 前缀
            # 例如: flight1_DJI_0001_V.JPG
            # dst_rgb_name = f"{item.flight_folder}_{item.rgb_name}"
            dst_rgb_name = item.rgb_name
            shutil.copy2(src_rgb, vis_dir / dst_rgb_name)
        else:
            print(f"警告: 找不到可见光文件 {src_rgb}")

        # --- 处理红外照片 ---
        if item.t_dir and item.t_name:
            src_t = Path(item.t_dir) / item.t_name
            if src_t.exists():
                # 保持前缀一致，方便后续一一对应
                # dst_t_name = f"{item.flight_folder}_{item.t_name}"
                dst_t_name = item.t_name
                shutil.copy2(src_t, thr_dir / dst_t_name)
            else:
                print(f"警告: 找不到红外文件 {src_t}")

    print("\n拷贝完成！")
    print(f"可见光照片数量: {len(list(vis_dir.glob('*.JPG')))}")
    print(f"红外照片数量: {len(list(thr_dir.glob('*.JPG')))}")

# endregion


# region read exif to json

def export_exifs(input_dir, output_path):
    image_list = os.listdir(input_dir)
    exif_dict = {}
    for image_name in tqdm(image_list, desc="Export exif"):
        image_path = os.path.join(input_dir, image_name)
        exif_dict[image_name] = pyexif_to_dict(image_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(exif_dict, f, ensure_ascii=False, indent=2)
    print(f"Export exif to {output_path}")

def pyexif_to_dict(img_path: str | Path) -> Dict[str, str]:
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()

# endregion


# region split data by manual order

def split_data_by_index_and_excel(excel_path, items_path, split_rgb_views_path, split_t_views_path):
    """
    excel_path: Excel 文件路径
    items: 你之前 build_index 得到的 List[ImageItem]
    output_root: 目标根目录
    """
    os.makedirs(split_rgb_views_path, exist_ok=True)
    os.makedirs(split_t_views_path, exist_ok=True)

    # 1. 读取 Excel
    df = pd.read_excel(excel_path)

    items = load_index(items_path)

    print("开始基于索引配对整理文件...")

    # 2. 遍历 Excel 的每一行（每一个 View）
    for _, row in df.iterrows():
        view_id = f'V{int(row["View id"]):02d}'
        start_name = str(row['start name'])
        end_name = str(row['end name'])
        
        # 创建子文件夹：visible 和 thermal
        target_v_folder = os.path.join(split_rgb_views_path, view_id)
        target_t_folder = os.path.join(split_t_views_path, view_id)
        os.makedirs(target_v_folder, exist_ok=True)
        os.makedirs(target_t_folder, exist_ok=True)
        
        count = 0
        # 3. 直接在索引表 (items) 中筛选
        for item in tqdm(items, desc=f"Processing View {view_id}", leave=False):
            # 获取不带后缀的 RGB 文件名进行比较
            rgb_stem = Path(item.rgb_name).stem
            
            # 核心判断逻辑
            if start_name <= rgb_stem <= end_name:
                # --- 拷贝可见光 ---
                src_rgb = os.path.join(item.rgb_dir, item.rgb_name)
                dst_rgb = os.path.join(target_v_folder, item.rgb_name)
                shutil.copy2(src_rgb, dst_rgb)
                
                # --- 拷贝对应的红外 (借助索引表里已经配对好的 t_name) ---
                if item.t_name and item.t_dir:
                    src_t = os.path.join(item.t_dir, item.t_name)
                    dst_t = os.path.join(target_t_folder, item.t_name)
                    if os.path.exists(src_t):
                        shutil.copy2(src_t, dst_t)
                
                count += 1
                
        print(f"View {view_id}: 已完成 {count} 组配对拷贝 (范围: {start_name} -> {end_name})")

    print("\n所有任务已完成！")

# endregion

if __name__ == '__main__':
    pass
    root_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data'
    docs_dir = os.path.join(root_dir, 'docs')

    source_dir = os.path.join(root_dir, 'collected data')
    index_path = os.path.join(root_dir, 'index.json')

    split_data = os.path.join(root_dir, 'split_data')
    split_rgb = os.path.join(split_data, 'visible')
    split_t = os.path.join(split_data, 'thermal')
    exif_rgb = split_rgb + '_exif.json'
    exif_t = split_t + '_exif.json'

    views_path = os.path.join(docs_dir, 'visible_views.xlsx')  # Excel 文件路径
    split_rgb_views = os.path.join(split_data, 'visible_views')
    split_t_views = os.path.join(split_data, 'thermal_views')

    split_rgb_views_map = os.path.join(docs_dir, 'visible_views_map.html')
    split_t_views_map = os.path.join(docs_dir, 'thermal_views_map.html')
    
    # step 1: build index for rgb-t pairs
    # build_index(source_dir, index_path)

    # step 2: copy rgb, t to different folder
    # export_dataset(index_path, split_data)

    # step 3: read exif to json
    # read_exifs(split_rgb, exif_rgb)
    # read_exifs(split_t, exif_t)

    # step 4: manually check
    
    # step 5: split data by manual order
    # split_data_by_index_and_excel(views_path, index_path, split_rgb_views, split_t_views)
    
    # step 6: build views map
    build_views_map(split_rgb_views, split_rgb_views_map, pick_method="middle")
    # build_views_map(split_t_views, split_t_views_map, pick_method="last")

    # root_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data'
    # excel_path = os.path.join(root_dir, 'visible_views.xlsx')  # Excel 文件路径
    # rgb_dir = os.path.join(root_dir, 'visible')    # 原始图像存放的文件夹路径
    # rgb_dir_split = os.path.join(root_dir, 'visible_views') # 目标根文件夹路径
    # t_dir = os.path.join(root_dir, 'thermal')
    # t_dir_split = os.path.join(root_dir, 'thermal_views')
    # split_data_by_manual_order(excel_path, rgb_dir, rgb_dir_split)
    # split_data_by_manual_order(excel_path, t_dir, t_dir_split)
