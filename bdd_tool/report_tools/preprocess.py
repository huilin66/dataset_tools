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

from direction_check import process_views_data

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


class FloorManager:
    def __init__(self, floor_params=None, cache_file=None):
        """
        :param floor_params: 用于构建的参数字典 (构建模式必填)
        :param cache_file: 用于保存或加载的 JSON 文件路径 (加载模式必填)
        """
        self.floor_map = {} 
        self.is_valid = False
        self.base_height = 0.0 # 用于绘图
        self.final_calc_height = 0.0 # 用于绘图

        # === 逻辑分支 ===
        # 情况 1: 提供了缓存文件，且文件存在 -> 直接加载 (Load Mode)
        if cache_file and os.path.exists(cache_file):
            print(f"📂 发现缓存文件 '{cache_file}'，正在加载...")
            self._load_from_file(cache_file)
        
        # 情况 2: 提供了参数 -> 重新构建 (Build Mode)
        elif floor_params:
            print("⚙️ 未找到缓存或强制构建，正在计算楼层...")
            self.params = floor_params
            self._parse_and_build()
            # 如果构建成功且指定了缓存路径，自动保存
            if self.is_valid and cache_file:
                self.write_floor_map(cache_file)
        
        # 情况 3: 既没文件也没参数 -> 报错
        else:
            raise ValueError("❌ 必须提供 floor_params 进行构建，或提供有效的 cache_file 进行加载。")

        # 打印图表 (可选，只有在数据完整时打印)
        if self.is_valid:
            self.print_floor_chart()

    def _parse_and_build(self):
        # 补丁：为了让 print_chart 不报错，这里模拟原逻辑的 scale 计算来获取 base_height
        p = self.params
        scale = 0.001 if p['normal floor height'] > 100 else 1.0
        self.base_height = p['base_height'] * scale
        
        
        base_h = p['base_height'] * scale
        final_h = p['final height'] * scale
        norm_h = p['normal floor height'] * scale
        
        # 转换列表和字典中的高度
        podium_hs = [h * scale for h in p['podium heights']]
        top_hs = [h * scale for h in p['top heights']]
        special_hs = {str(k): v * scale for k, v in p['special heights'].items()}
        
        # 2. 构建楼层序列 (Name, Height)
        floor_sequence = []
        
        # A. Podium (裙楼/底层)
        if len(p['podium names']) != len(podium_hs):
            print(f"❌ 楼层参数错误: Podium 名字数量 ({len(p['podium names'])}) 与 高度数量 ({len(podium_hs)}) 不一致")
            return
            
        for name, h in zip(p['podium names'], podium_hs):
            floor_sequence.append((str(name), h))
            
        # B. Normal (标准层 + 特殊层)
        # range 是左闭右闭，所以 end + 1
        start_idx, end_idx = p['normal height number list']
        expected_norm_count = p['normal height numbers']
        
        # 校验数量
        real_norm_count = end_idx - start_idx + 1
        if real_norm_count != expected_norm_count:
             print(f"⚠️ 警告: Normal floor 数量定义不一致 (Number: {expected_norm_count} vs List range: {real_norm_count})，以 List 为准")

        for i in range(start_idx, end_idx + 1):
            name = str(i)
            # 检查是否是特殊层
            h = special_hs.get(name, norm_h)
            floor_sequence.append((name, h))
            
        # C. Top (顶层)
        if len(p['top names']) != len(top_hs):
            print(f"❌ 楼层参数错误: Top 名字数量 ({len(p['top names'])}) 与 高度数量 ({len(top_hs)}) 不一致")
            return

        for name, h in zip(p['top names'], top_hs):
            floor_sequence.append((str(name), h))

        # 3. 生成高度分布字典 & 校验总高度
        current_z = base_h
        
        for name, h in floor_sequence:
            # 格式化 Key: "楼层编号/F"
            key = f"{name}/F"
            self.floor_map[key] = (current_z, current_z + h)
            current_z += h
            
        # 4. 校验高度闭环
        # 理论总高度 = final - base
        # 累加总高度 = current_z - base_h
        self.final_calc_height = current_z
        diff = abs(current_z - final_h)
        
        print(f"🏢 楼层构建完成: 起始 {base_h:.2f}m -> 计算结束 {current_z:.2f}m (定义结束 {final_h:.2f}m)")
        
        if diff > 0.1: # 允许 10cm 误差
            print(f"⚠️ 警告: 建筑高度校验失败! 偏差 {diff:.4f}m")
            print("   请检查: base_height, final height 或 各层高度之和是否匹配")
        else:
            print("✅ 建筑高度校验通过")
            self.is_valid = True

    def _load_from_file(self, path):
        """从 JSON 加载数据，恢复状态"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # JSON 读取后的格式通常是:
            # { "meta": {"base": 10.0, "top": 100.0}, "map": {"1/F": [10, 13], ...} }
            # 为了简单，如果你只存了 map，就只读 map。
            # 但为了 print_chart 能用，建议保存时多存一点元数据。
            
            if "floor_map" in data:
                self.floor_map = data["floor_map"]
                self.base_height = data.get("base_height", 0.0)
                self.final_calc_height = data.get("final_calc_height", 0.0)
            else:
                # 兼容旧版本只存了 map 的情况
                self.floor_map = data
            
            self.is_valid = True
            print("✅ 楼层数据加载成功")
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            self.is_valid = False

    def write_floor_map(self, output_path):
        """将楼层映射及元数据写入 JSON 文件"""
        # 建议把元数据一起存了，这样下次加载后还能画图
        save_data = {
            "base_height": self.base_height,
            "final_calc_height": self.final_calc_height,
            "floor_map": self.floor_map
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"💾 楼层数据已保存至: {output_path}")

    def get_floor(self, z_value):
        # 原逻辑保持不变
        # 注意：JSON 加载回来的 value 是 List [start, end]
        # 但 Python 的解包赋值 (start, end) = [10, 13] 对 List 和 Tuple 都适用
        # 所以这里的代码完全不需要改动
        epsilon = 0.01 
        for name, (start, end) in self.floor_map.items():
            if start - epsilon <= z_value < end + epsilon:
                return name

        # 如果找不到
        sorted_floors = sorted(self.floor_map.values(), key=lambda x: x[0])
        if not sorted_floors: return "Unknown"
        
        min_h = sorted_floors[0][0]
        max_h = sorted_floors[-1][1]
        
        if z_value < min_h:
            return "Below Base"
        elif z_value >= max_h:
            return "Above Top"
        
        return "Unknown"

    def print_floor_chart(self):
        # 稍微修改，不再依赖 self.params，而是依赖 self.base_height
        if not self.floor_map: return

        print("\n🏢 Building Elevation Chart (Top-Down)")
        print("=" * 40)
        print(f"{'[TOP]':<10} ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅  {self.final_calc_height:7.2f}m")

        sorted_floors = sorted(self.floor_map.items(), key=lambda item: item[1][0], reverse=True)
        for name, (start_z, end_z) in sorted_floors:
            print(f"{name:<10} ______  {start_z:7.2f}m")
            
        # 这里改用 self.base_height
        print(f"{'[BASE]':<10} ______  {self.base_height:7.2f}m")
        print("=" * 40 + "\n")

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

    build_height_path = os.path.join(docs_dir, 'build_heights.json')
    
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
    # process_views_data(split_rgb_views, docs_dir, pick_method="middle")

    # step 7: get build elevation chart
    # floor_param = {
    #     'base_height':22500,
    #     'final height':123800,
    #     'normal floor height':3150,
    #     'podium heights': [6000, 5000, 4500, 5500],
    #     'top heights': [6650],
    #     'podium names': ['LG', 'G', '1', '2'],
    #     'top names': ['ROOF'],
    #     'normal height numbers': 23,
    #     'normal height number list': [3, 25],
    #     'special heights': {
    #         '4': 3450,
    #         '11': 3450,
    #         '18': 3450,
    #         '23': 3450,
    #     }
    # }
    # fm = FloorManager(floor_params=floor_param, cache_file=build_height_path)
    # fm = FloorManager(cache_file=build_height_path)


