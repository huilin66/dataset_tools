"""Image file helpers."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def matching_label_path(labels_dir: str | Path, image_path: str | Path) -> Path:
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"
