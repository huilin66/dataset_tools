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


if __name__ == "__main__":
    pass
    from pprint import pprint
    image_path = r"E:\data\thesis\HTM\collected data\DJI_202512161540_008_filter\DJI_20251216155812_0537_V.JPG"
    params = parse_dji_xmp(image_path)
    pprint(params)
