"""Convert parsed detections to YOLO labels."""

from __future__ import annotations

from classes import resolve_class_id
from parser import ParsedDetection
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
