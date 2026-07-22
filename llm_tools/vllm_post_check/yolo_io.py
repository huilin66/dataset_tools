"""YOLO txt I/O and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None

    def as_line(self, include_conf: bool = True) -> str:
        values: list[str] = [
            str(int(self.class_id)),
            f"{self.x_center:.6f}",
            f"{self.y_center:.6f}",
            f"{self.width:.6f}",
            f"{self.height:.6f}",
        ]
        if include_conf and self.confidence is not None:
            values.append(f"{self.confidence:.6f}")
        return " ".join(values)


def read_yolo_txt(path: str | Path) -> list[YoloBox]:
    path = Path(path)
    if not path.exists():
        return []
    boxes: list[YoloBox] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"Invalid YOLO row {path}:{line_no}: {line.strip()}")
            confidence = float(parts[5]) if len(parts) >= 6 else None
            boxes.append(
                YoloBox(
                    class_id=int(float(parts[0])),
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                    confidence=confidence,
                )
            )
    return boxes


def write_yolo_txt(path: str | Path, boxes: list[YoloBox], include_conf: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for box in boxes:
            f.write(box.as_line(include_conf=include_conf) + "\n")


def yolo_to_xyxy(box: YoloBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    cx = box.x_center * image_width
    cy = box.y_center * image_height
    bw = box.width * image_width
    bh = box.height * image_height
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def xyxy_to_yolo(
    class_id: int,
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    confidence: float | None = None,
) -> YoloBox:
    x1, y1, x2, y2 = clamp_xyxy(xyxy, image_width, image_height)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return YoloBox(
        class_id=class_id,
        x_center=((x1 + x2) / 2) / image_width,
        y_center=((y1 + y2) / 2) / image_height,
        width=bw / image_width,
        height=bh / image_height,
        confidence=confidence,
    )


def clamp_xyxy(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = min(max(float(x1), 0.0), float(image_width))
    y1 = min(max(float(y1), 0.0), float(image_height))
    x2 = min(max(float(x2), 0.0), float(image_width))
    y2 = min(max(float(y2), 0.0), float(image_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def box_area(xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def nms_yolo(boxes: list[YoloBox], image_width: int, image_height: int, iou_threshold: float) -> list[YoloBox]:
    ordered = sorted(boxes, key=lambda b: b.confidence if b.confidence is not None else 1.0, reverse=True)
    kept: list[YoloBox] = []
    for box in ordered:
        xyxy = yolo_to_xyxy(box, image_width, image_height)
        should_drop = False
        for kept_box in kept:
            if kept_box.class_id != box.class_id:
                continue
            if iou(xyxy, yolo_to_xyxy(kept_box, image_width, image_height)) > iou_threshold:
                should_drop = True
                break
        if not should_drop:
            kept.append(box)
    return kept
