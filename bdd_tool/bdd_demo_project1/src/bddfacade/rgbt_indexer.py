from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json

RGB_SUFFIX = "_V.JPG"
T_SUFFIX = "_T.JPG"


@dataclass
class ImageItem:
    id: str
    flight_folder: str
    rgb_path: str
    t_path: Optional[str] = None


def is_rgb(name: str) -> bool:
    return name.upper().endswith(RGB_SUFFIX)

def rgb_to_t_name(rgb_name: str) -> str:
    # DJI_..._V.JPG -> DJI_..._T.JPG
    return rgb_name[:-len(RGB_SUFFIX)] + T_SUFFIX


def build_index(data_root: Path) -> List[ImageItem]:
    """
    扫描 data_root 下所有子目录（航线文件夹），收集 *_V.JPG 并配对 *_T.JPG。
    """
    items: List[ImageItem] = []
    idx = 0

    for folder in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        flight_folder = folder.name
        # 只看当前文件夹内的文件
        files = {p.name: p for p in folder.iterdir() if p.is_file()}
        for name, p in tqdm(sorted(files.items()), desc=f"Scan {flight_folder} to build index for RGBT pairs"):
            if not is_rgb(name):
                continue
            t_name = rgb_to_t_name(name)
            t_path = str(files[t_name]) if t_name in files else None
            items.append(
                ImageItem(
                    id=f"{flight_folder}:{idx:06d}",
                    flight_folder=flight_folder,
                    rgb_path=str(p),
                    t_path=t_path,
                )
            )
            idx += 1
    
    print('scan info:')
    print(f'total {len(items)} images')
    print(f'total {sum(1 for x in items if x.t_path)} pairs')
    return items


def save_index(items: List[ImageItem], out_path: Path) -> None:
    out = [asdict(x) for x in items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'save {out_path}')

def load_index(path: Path) -> List[ImageItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ImageItem(**x) for x in data]
