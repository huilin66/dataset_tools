from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Optional


_RDF_START = b"<rdf:Description "
_RDF_END = b"</rdf:Description>"

# 抽取 drone-dji:Key="Value"
_DJI_ATTR_RE = re.compile(r'drone-dji:(?P<key>[A-Za-z0-9_]+)\s*=\s*"(?P<val>[^"]*)"')

# 抽取普通 XML 属性 Key="Value"（如果你未来也想拿别的命名空间）
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z0-9_:\-]+)\s*=\s*"(?P<val>[^"]*)"')


def extract_rdf_description_bytes(jpg_path: Path) -> Optional[bytes]:
    """
    从 JPG 文件中提取第一个 <rdf:Description ... </rdf:Description> 区块（bytes）。
    """
    data = jpg_path.read_bytes()
    s = data.find(_RDF_START)
    if s < 0:
        return None
    e = data.find(_RDF_END, s)
    if e < 0:
        return None
    e = e + len(_RDF_END)
    return data[s:e]


def parse_dji_xmp(jpg_path: str | Path) -> Dict[str, str]:
    """
    解析 DJI Matrice 4T 写入 JPG 的 XMP 扩展元数据（drone-dji:*）。
    返回 dict：key -> value（字符串）
    """
    p = Path(jpg_path)
    block = extract_rdf_description_bytes(p)
    if not block:
        return {}

    # 尽量用 UTF-8/ASCII 解码（XMP 通常是 UTF-8）
    text = block.decode("utf-8", errors="ignore")

    out: Dict[str, str] = {}
    for m in _DJI_ATTR_RE.finditer(text):
        out[m.group("key")] = m.group("val")
    return out


def parse_all_attrs_in_rdf(jpg_path: str | Path) -> Dict[str, str]:
    """
    如需调试：把 rdf:Description 里所有属性都抓出来（包括 drone-dji:）
    """
    p = Path(jpg_path)
    block = extract_rdf_description_bytes(p)
    if not block:
        return {}
    text = block.decode("utf-8", errors="ignore")
    out: Dict[str, str] = {}
    for m in _ATTR_RE.finditer(text):
        out[m.group("key")] = m.group("val")
    return out

import os
from PIL import Image, ExifTags

def debug_exif(img_path):
    img = Image.open(img_path)
    exif = img.getexif()
    if not exif:
        print(f"No EXIF found in {img_path}")
        return

    # 建立 Tag ID 到名称的映射
    tag_map = {v: k for k, v in ExifTags.TAGS.items()}
    
    print(f"--- EXIF for {os.path.basename(img_path)} ---")
    
    # 读取型号
    model = exif.get(272) # 272 is Model
    print(f"Model: {model}")

    # 读取焦距
    focal = exif.get(37386) # 37386 is FocalLength
    print(f"Focal Length: {focal} mm")
    
    # 35mm 等效 (参考用)
    focal_35 = exif.get(41989)
    print(f"35mm Equivalent: {focal_35} mm")

def pyexif_to_dict(img_path: str | Path) -> Dict[str, str]:
    import pyexif
    img = pyexif.ExifEditor(img_path)
    return img.getDictTags()

if __name__ == "__main__":
    pass
    # params = parse_dji_xmp(image_path)
    # pprint(params)
    # print('----------------')
    # debug_exif(image_path)
    # print('----------------')
    print('----------------RGB EXIF----------------')
    from pprint import pprint
    image_path = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\visible\DJI_20251216155617_0520_V.JPG"
    exif_dict = pyexif_to_dict(image_path)
    pprint(exif_dict)
    # print('----------------Thermal EXIF----------------')
    # image_path = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\thermal\DJI_20251216155618_0520_T.JPG'
    # exif_dict = pyexif_to_dict(image_path)
    # pprint(exif_dict)