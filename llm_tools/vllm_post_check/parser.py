"""Parse Qwen/VLLM JSON responses into normalized detections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from yolo_io import clamp_xyxy


@dataclass
class ParsedDetection:
    class_name: str
    xyxy: tuple[float, float, float, float] | None
    confidence: float | None
    keep: bool = True


def extract_json(text: str) -> Any:
    text = str(text or "").strip()
    if not text:
        return {"detections": []}
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([idx for idx in [text.find("{"), text.find("[")] if idx >= 0], default=-1)
        if start < 0:
            raise
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        return json.loads(text[start:end + 1])


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "reject", "drop"}:
        return False
    if text in {"true", "1", "yes", "keep"}:
        return True
    return default


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_from_item(item: dict[str, Any]) -> tuple[Any, str]:
    if "bbox_norm_1000" in item:
        return item["bbox_norm_1000"], "norm1000"
    for key in ("bbox", "box", "bbox_2d", "xyxy"):
        if key in item:
            return item[key], "auto"
    return None, "auto"


def _normalize_bbox(raw: Any, image_width: int, image_height: int, mode: str) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    vals = [float(raw[idx]) for idx in range(4)]
    max_val = max(vals)
    if mode == "norm1000":
        vals = [
            vals[0] / 1000 * image_width,
            vals[1] / 1000 * image_height,
            vals[2] / 1000 * image_width,
            vals[3] / 1000 * image_height,
        ]
    elif max_val <= 1.5:
        vals = [
            vals[0] * image_width,
            vals[1] * image_height,
            vals[2] * image_width,
            vals[3] * image_height,
        ]
    return clamp_xyxy((vals[0], vals[1], vals[2], vals[3]), image_width, image_height)


def parse_detections(response_text: str, image_width: int, image_height: int, require_bbox: bool = True) -> list[ParsedDetection]:
    payload = extract_json(response_text)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("detections") or payload.get("objects") or payload.get("results") or []
    else:
        items = []
    detections: list[ParsedDetection] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        keep = _parse_bool(item.get("keep", True), default=True)
        class_name = str(item.get("class_name") or item.get("class") or item.get("label") or "").strip()
        confidence = _parse_float(item.get("confidence", item.get("score", None)))
        raw_bbox, mode = _bbox_from_item(item)
        xyxy = _normalize_bbox(raw_bbox, image_width, image_height, mode)
        if require_bbox and xyxy is None:
            continue
        detections.append(ParsedDetection(class_name=class_name, xyxy=xyxy, confidence=confidence, keep=keep))
    return detections
