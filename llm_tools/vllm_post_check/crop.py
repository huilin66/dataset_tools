"""Crop YOLO candidates and map crop predictions back to the original image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from yolo_io import YoloBox, clamp_xyxy, yolo_to_xyxy


@dataclass
class CropItem:
    image_path: Path
    crop_path: Path
    source_box: YoloBox
    source_xyxy: tuple[float, float, float, float]
    crop_xyxy: tuple[float, float, float, float]
    crop_size: tuple[int, int]


def expand_xyxy(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    return clamp_xyxy((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_width, image_height)


def create_crop(
    image_path: str | Path,
    box: YoloBox,
    crop_path: str | Path,
    padding_ratio: float = 0.15,
) -> CropItem:
    image_path = Path(image_path)
    crop_path = Path(crop_path)
    with Image.open(image_path) as image:
        width, height = image.size
        source_xyxy = clamp_xyxy(yolo_to_xyxy(box, width, height), width, height)
        crop_xyxy = expand_xyxy(source_xyxy, width, height, padding_ratio)
        ix1, iy1, ix2, iy2 = [int(round(v)) for v in crop_xyxy]
        crop = image.crop((ix1, iy1, ix2, iy2))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
        crop_size = crop.size
    return CropItem(
        image_path=image_path,
        crop_path=crop_path,
        source_box=box,
        source_xyxy=source_xyxy,
        crop_xyxy=(float(ix1), float(iy1), float(ix2), float(iy2)),
        crop_size=crop_size,
    )


def crop_xyxy_to_image(
    xyxy: tuple[float, float, float, float],
    crop_item: CropItem,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    offset_x, offset_y = crop_item.crop_xyxy[0], crop_item.crop_xyxy[1]
    x1, y1, x2, y2 = xyxy
    return clamp_xyxy((x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y), image_width, image_height)
