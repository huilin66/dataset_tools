"""Convert parsed detections to YOLO labels."""

from __future__ import annotations

from classes import resolve_class_id
from parser import ParsedDetection, ParsedP2GroundingDetection
from yolo_io import YoloBox, nms_yolo, xyxy_to_yolo


def detections_to_yolo(
    detections: list[ParsedDetection],
    classes: list[str],
    image_width: int,
    image_height: int,
    min_conf: float = 0.0,
    iou_threshold: float = 0.5,
) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for det in detections:
        if not det.keep or det.xyxy is None:
            continue
        if det.confidence is not None and det.confidence < min_conf:
            continue
        class_id = resolve_class_id(det.class_name, classes)
        if class_id is None:
            continue
        box = xyxy_to_yolo(class_id, det.xyxy, image_width, image_height, det.confidence)
        if box.width <= 0 or box.height <= 0:
            continue
        boxes.append(box)
    return nms_yolo(boxes, image_width, image_height, iou_threshold)


def p2_detections_to_yolo(
    detections: list[ParsedP2GroundingDetection],
    classes: list[str],
    image_width: int,
    image_height: int,
    min_conf: float = 0.0,
    iou_threshold: float = 0.5,
) -> list[YoloBox]:
    """Convert p2 normalized-1000 bbox_2d directly to YOLO normalized xywh."""
    boxes: list[YoloBox] = []
    for det in detections:
        if det.confidence is not None and det.confidence < min_conf:
            continue
        class_id = resolve_class_id(det.class_name, classes)
        if class_id is None:
            continue
        x1, y1, x2, y2 = det.bbox_norm_1000
        box = YoloBox(
            class_id=class_id,
            x_center=(x1 + x2) / 2000.0,
            y_center=(y1 + y2) / 2000.0,
            width=(x2 - x1) / 1000.0,
            height=(y2 - y1) / 1000.0,
            confidence=det.confidence,
        )
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    return nms_yolo(boxes, image_width, image_height, iou_threshold)


def p2_classification_to_yolo(
    class_name: str,
    source_box: YoloBox,
    classes: list[str],
    fallback_confidence: float | None,
) -> YoloBox | None:
    """Preserve a YOLO candidate box and replace its class from p2 crop output."""
    class_id = resolve_class_id(class_name, classes)
    if class_id is None:
        return None
    return YoloBox(
        class_id=class_id,
        x_center=source_box.x_center,
        y_center=source_box.y_center,
        width=source_box.width,
        height=source_box.height,
        confidence=(
            source_box.confidence
            if source_box.confidence is not None
            else fallback_confidence
        ),
    )
