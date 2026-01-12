from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import json
import os
from pathlib import Path
import re
import shutil
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# region rgb-t index

@dataclass
class ImageItem:
    id: str
    flight_folder: str
    rgb_dir: str
    rgb_name: str
    t_dir: Optional[str] = None
    t_name: Optional[str] = None


def build_index(data_path, index_path, standard_DJI_format=False) -> List[ImageItem]:
    """
    改进版：基于序列号(Sequence Number)配对，解决时间戳秒数偏差问题。
    """
    if not standard_DJI_format:
        raise NotImplementedError("Only support standard DJI format.")

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

def _copy_item_worker(item, vis_dir: Path, thr_dir: Path):
    """
    单个 Item 的拷贝逻辑（提取出来供单线程和多线程共用）
    """
    # --- 处理可见光照片 ---
    src_rgb = Path(item.rgb_dir) / item.rgb_name
    if src_rgb.exists():
        dst_rgb_name = item.rgb_name
        # 注意：多线程下 shutil.copy2 是线程安全的（只要目标文件不同）
        shutil.copy2(src_rgb, vis_dir / dst_rgb_name)
    else:
        # 使用 tqdm.write 代替 print，防止在多线程+进度条时控制台输出错乱
        tqdm.write(f"警告: 找不到可见光文件 {src_rgb}")

    # --- 处理红外照片 ---
    if item.t_dir and item.t_name:
        src_t = Path(item.t_dir) / item.t_name
        if src_t.exists():
            dst_t_name = item.t_name
            shutil.copy2(src_t, thr_dir / dst_t_name)
        else:
            tqdm.write(f"警告: 找不到红外文件 {src_t}")


def export_single_modal_data(index_path: str, output_root: str, num_workers: int = 1):
    """
    将所有数据拷贝到指定文件夹下的 visible 和 thermal 目录
    
    Args:
        index_path: 索引文件路径
        output_root: 输出根目录
        num_workers: 线程数。1 为单线程（默认），>1 为多线程。
    """
    items = load_index(Path(index_path))

    # 1. 准备目标路径
    out_path = Path(output_root)
    vis_dir = out_path / "visible"
    thr_dir = out_path / "thermal"

    # 递归创建目录（如果不存在）
    vis_dir.mkdir(parents=True, exist_ok=True)
    thr_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始拷贝数据到: {out_path}")
    print(f"模式: {'多线程并行 (' + str(num_workers) + ' workers)' if num_workers > 1 else '单线程顺序执行'}")

    # 2. 绑定固定参数 (vis_dir, thr_dir)，只留 item 作为变参
    # 这样我们可以方便地把 worker 函数传给 map
    worker_func = partial(_copy_item_worker, vis_dir=vis_dir, thr_dir=thr_dir)

    # 3. 执行拷贝
    if num_workers > 1:
        # --- 多线程模式 ---
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # executor.map 会按顺序启动任务，tqdm 负责包装这个生成器来显示进度
            list(tqdm(executor.map(worker_func, items), total=len(items), desc="Copying images (Multi)"))
    else:
        # --- 单线程模式 (保留原有逻辑) ---
        for item in tqdm(items, desc="Copying images (Single)"):
            worker_func(item)

    print("\n拷贝完成！")
    # 统计数量 (glob 是磁盘操作，可能会花一点时间，但在结束时执行无妨)
    print(f"可见光照片数量: {len(list(vis_dir.glob('*.JPG')))}")
    print(f"红外照片数量: {len(list(thr_dir.glob('*.JPG')))}")
    
# endregion


# region split data by manual order

def _copy_worker(item, target_v_folder, target_t_folder):
    """
    单个文件的拷贝任务（将在多线程中运行）
    返回 1 表示成功拷贝一组，0 表示失败或未找到
    """
    success = False
    
    # --- 拷贝可见光 ---
    src_rgb = os.path.join(item.rgb_dir, item.rgb_name)
    dst_rgb = os.path.join(target_v_folder, item.rgb_name)
    
    if os.path.exists(src_rgb):
        shutil.copy2(src_rgb, dst_rgb)
        success = True
    
    # --- 拷贝红外 ---
    if item.t_name and item.t_dir:
        src_t = os.path.join(item.t_dir, item.t_name)
        dst_t = os.path.join(target_t_folder, item.t_name)
        if os.path.exists(src_t):
            shutil.copy2(src_t, dst_t)
            
    return 1 if success else 0

def split_data_by_index(index_path, items_path, split_rgb_views_path, split_t_views_path, num_workers=8):
    """
    num_workers: 控制并行拷贝的线程数（建议 4-16）
    """
    os.makedirs(split_rgb_views_path, exist_ok=True)
    os.makedirs(split_t_views_path, exist_ok=True)

    # 1. 读取数据
    df = pd.read_excel(index_path)
    items = load_index(items_path)
    
    print(f"开始整理文件 (View 顺序执行, 内部拷贝采用 {num_workers} 线程并行)...")

    # 创建一个线程池（在整个大循环外创建，复用线程，减少开销）
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        
        # 2. 遍历 Excel 的每一行（保持串行，以便显示清晰的进度条）
        for _, row in df.iterrows():
            view_id = f'V{int(row["View id"]):02d}'
            start_name = str(row['start name'])
            end_name = str(row['end name'])
            
            # 准备目标文件夹
            target_v_folder = os.path.join(split_rgb_views_path, view_id)
            target_t_folder = os.path.join(split_t_views_path, view_id)
            os.makedirs(target_v_folder, exist_ok=True)
            os.makedirs(target_t_folder, exist_ok=True)
            
            # --- 步骤 A: 筛选 (Search Phase) ---
            # 先找出属于这个 View 的所有 item，这个过程是纯内存比较，非常快，不需要多线程
            matched_items = []
            for item in items:
                rgb_stem = Path(item.rgb_name).stem
                if start_name <= rgb_stem <= end_name:
                    matched_items.append(item)
            
            if not matched_items:
                print(f"View {view_id}: 未找到匹配文件 ({start_name} -> {end_name})")
                continue

            # --- 步骤 B: 并行拷贝 (Copy Phase) ---
            # 使用 executor.submit 并发执行拷贝，确保进度条能更新
            futures = []
            for item in matched_items:
                futures.append(executor.submit(_copy_worker, item, target_v_folder, target_t_folder))
            
            # 使用 tqdm 显示进度
            for future in tqdm(futures, desc=f"View {view_id} Copying", unit="img"):
                future.result()  # 等待每个任务完成并获取结果

    print("\n所有任务已完成！")

# endregion