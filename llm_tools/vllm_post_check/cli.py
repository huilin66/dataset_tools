"""CLI for simple Qwen/VLLM YOLO post-check workflows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from classes import load_classes, resolve_class_id
from crop import create_crop, crop_xyxy_to_image
from image_io import image_size, iter_images, matching_label_path
from parser import ParsedDetection, parse_detections
from postprocess import detections_to_yolo
from prompts import crop_refine_prompt, full_image_prompt
from vllm_client import VLLMClient
from yolo_io import YoloBox, nms_yolo, read_yolo_txt, write_yolo_txt, xyxy_to_yolo

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency fallback
    tqdm = None


def _write_raw(path: Path, response_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response_text, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _manifest_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=["image", "label", "result_json", "raw", "crops", "input_boxes", "output_boxes", "status", "message"],
    )
    writer.writeheader()
    return f, writer


def _write_manifest_row(manifest, manifest_file, row: dict) -> None:
    manifest.writerow(row)
    manifest_file.flush()


def _progress(items: list[Path], desc: str, no_progress: bool) -> Iterable[Path]:
    if no_progress:
        return items
    if tqdm is None:
        print("tqdm is not installed; install it with: pip install tqdm")
        return items
    return tqdm(items, desc=desc, unit="img")


def _set_progress_postfix(progress, **values) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(**values)


def _box_to_dict(box: YoloBox) -> dict:
    return {
        "class_id": int(box.class_id),
        "x_center": box.x_center,
        "y_center": box.y_center,
        "width": box.width,
        "height": box.height,
        "confidence": box.confidence,
    }


def _parsed_to_dict(det: ParsedDetection) -> dict:
    return {
        "class_name": det.class_name,
        "bbox_xyxy": list(det.xyxy) if det.xyxy is not None else None,
        "confidence": det.confidence,
        "keep": det.keep,
    }


def build_client(args) -> VLLMClient:
    return VLLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def run_full_image(args) -> None:
    classes = load_classes(args.classes)
    client = build_client(args)
    images = iter_images(args.images, recursive=not args.no_recursive)
    if args.limit:
        images = images[: args.limit]
    output = Path(args.output)
    labels_dir = output / "labels"
    raw_dir = output / "raw"
    results_dir = output / "results"
    manifest_file, manifest = _manifest_writer(output / "manifest.csv")
    progress = _progress(images, "full-image", args.no_progress)
    try:
        for image_path in progress:
            label_path = labels_dir / f"{image_path.stem}.txt"
            raw_path = raw_dir / f"{image_path.stem}.json"
            result_path = results_dir / f"{image_path.stem}.json"
            if label_path.exists() and raw_path.exists() and result_path.exists() and not args.overwrite:
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "raw": raw_path,
                    "status": "skipped",
                })
                _set_progress_postfix(progress, status="skipped")
                continue
            result = {
                "mode": "full-image",
                "image": str(image_path),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "status": "ok",
                "message": "",
                "classes_path": str(args.classes),
                "model": args.model,
                "base_url": args.base_url,
                "task_type": args.task_type,
                "detections": [],
                "output_boxes": [],
            }
            try:
                width, height = image_size(image_path)
                result["image_size"] = {"width": width, "height": height}
                response_text = client.predict(image_path, full_image_prompt(classes, args.task_type))
                _write_raw(raw_path, response_text)
                detections = parse_detections(response_text, width, height, require_bbox=True)
                boxes = detections_to_yolo(detections, classes, width, height, args.min_conf, args.iou)
                write_yolo_txt(label_path, boxes, include_conf=not args.no_conf)
                result["detections"] = [_parsed_to_dict(det) for det in detections]
                result["output_boxes"] = [_box_to_dict(box) for box in boxes]
                _write_json(result_path, result)
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "raw": raw_path,
                    "input_boxes": 0,
                    "output_boxes": len(boxes),
                    "status": "ok",
                })
                _set_progress_postfix(progress, status="ok", boxes=len(boxes))
            except Exception as exc:
                write_yolo_txt(label_path, [], include_conf=not args.no_conf)
                result["status"] = "error"
                result["message"] = str(exc)
                _write_json(result_path, result)
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "raw": raw_path,
                    "status": "error",
                    "message": str(exc),
                })
                _set_progress_postfix(progress, status="error")
    finally:
        manifest_file.close()


def run_crop_refine(args) -> None:
    classes = load_classes(args.classes)
    client = build_client(args)
    images = iter_images(args.images, recursive=not args.no_recursive)
    if args.limit:
        images = images[: args.limit]
    output = Path(args.output)
    labels_dir = output / "labels"
    raw_dir = output / "raw"
    crops_dir = output / "crops"
    results_dir = output / "results"
    manifest_file, manifest = _manifest_writer(output / "manifest.csv")
    progress = _progress(images, "crop-refine", args.no_progress)
    try:
        for image_path in progress:
            pred_path = matching_label_path(args.pred_labels, image_path)
            label_path = labels_dir / f"{image_path.stem}.txt"
            result_path = results_dir / f"{image_path.stem}.json"
            if label_path.exists() and result_path.exists() and not args.overwrite:
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "status": "skipped",
                })
                _set_progress_postfix(progress, status="skipped")
                continue

            result = {
                "mode": "crop-refine",
                "refine_mode": args.mode,
                "image": str(image_path),
                "pred_label_path": str(pred_path),
                "label_path": str(label_path),
                "result_json": str(result_path),
                "status": "ok",
                "message": "",
                "classes_path": str(args.classes),
                "model": args.model,
                "base_url": args.base_url,
                "task_type": args.task_type,
                "crop_padding": args.crop_padding,
                "input_boxes": [],
                "crops": [],
                "output_boxes": [],
            }

            if not pred_path.exists():
                write_yolo_txt(label_path, [], include_conf=not args.no_conf)
                result["status"] = "no_pred"
                _write_json(result_path, result)
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "input_boxes": 0,
                    "output_boxes": 0,
                    "status": "no_pred",
                })
                _set_progress_postfix(progress, status="no_pred", boxes=0)
                continue

            input_boxes = read_yolo_txt(pred_path)
            result["input_boxes"] = [_box_to_dict(box) for box in input_boxes]
            width, height = image_size(image_path)
            result["image_size"] = {"width": width, "height": height}
            output_boxes: list[YoloBox] = []
            crop_count = 0
            try:
                for idx, source_box in enumerate(input_boxes):
                    candidate_class = classes[source_box.class_id] if 0 <= source_box.class_id < len(classes) else str(source_box.class_id)
                    crop_path = crops_dir / image_path.stem / f"{idx:04d}_{candidate_class.replace(' ', '_')}.jpg"
                    crop_item = create_crop(image_path, source_box, crop_path, padding_ratio=args.crop_padding)
                    crop_count += 1
                    raw_path = raw_dir / image_path.stem / f"{idx:04d}.json"
                    crop_record = {
                        "index": idx,
                        "candidate_class": candidate_class,
                        "source_box": _box_to_dict(source_box),
                        "source_xyxy": list(crop_item.source_xyxy),
                        "crop_xyxy": list(crop_item.crop_xyxy),
                        "crop_size": {"width": crop_item.crop_size[0], "height": crop_item.crop_size[1]},
                        "crop_path": str(crop_path),
                        "raw_path": str(raw_path),
                        "detections": [],
                        "kept": False,
                        "message": "",
                    }
                    try:
                        response_text = client.predict(crop_path, crop_refine_prompt(classes, candidate_class, args.task_type))
                        _write_raw(raw_path, response_text)
                        crop_width, crop_height = crop_item.crop_size
                        parsed = parse_detections(response_text, crop_width, crop_height, require_bbox=args.mode == "detect")
                        crop_record["detections"] = [_parsed_to_dict(det) for det in parsed]
                        kept = [det for det in parsed if det.keep]
                        if not kept:
                            result["crops"].append(crop_record)
                            continue
                        if args.mode == "classification":
                            det = kept[0]
                            class_name = det.class_name or candidate_class
                            class_id = resolve_class_id(class_name, classes) if class_name else source_box.class_id
                            if class_id is None:
                                class_id = source_box.class_id
                            confidence = det.confidence if det.confidence is not None else source_box.confidence
                            if confidence is not None and confidence < args.min_conf:
                                result["crops"].append(crop_record)
                                continue
                            output_box = xyxy_to_yolo(class_id, crop_item.source_xyxy, width, height, confidence)
                            output_boxes.append(output_box)
                            crop_record["kept"] = True
                            crop_record["output_boxes"] = [_box_to_dict(output_box)]
                        else:
                            crop_outputs = []
                            for det in kept:
                                if det.xyxy is None:
                                    continue
                                mapped = crop_xyxy_to_image(det.xyxy, crop_item, width, height)
                                class_id = resolve_class_id(det.class_name, classes)
                                if class_id is None:
                                    continue
                                if det.confidence is not None and det.confidence < args.min_conf:
                                    continue
                                output_box = xyxy_to_yolo(class_id, mapped, width, height, det.confidence)
                                output_boxes.append(output_box)
                                crop_outputs.append(_box_to_dict(output_box))
                            crop_record["kept"] = bool(crop_outputs)
                            crop_record["output_boxes"] = crop_outputs
                    except Exception as crop_exc:
                        crop_record["message"] = str(crop_exc)
                    result["crops"].append(crop_record)

                output_boxes = nms_yolo(output_boxes, width, height, args.iou)
                write_yolo_txt(label_path, output_boxes, include_conf=not args.no_conf)
                result["output_boxes"] = [_box_to_dict(box) for box in output_boxes]
                _write_json(result_path, result)
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "crops": crop_count,
                    "input_boxes": len(input_boxes),
                    "output_boxes": len(output_boxes),
                    "status": "ok",
                })
                _set_progress_postfix(progress, status="ok", crops=crop_count, boxes=len(output_boxes))
            except Exception as exc:
                write_yolo_txt(label_path, [], include_conf=not args.no_conf)
                result["status"] = "error"
                result["message"] = str(exc)
                result["output_boxes"] = []
                _write_json(result_path, result)
                _write_manifest_row(manifest, manifest_file, {
                    "image": image_path,
                    "label": label_path,
                    "result_json": result_path,
                    "crops": crop_count,
                    "input_boxes": len(input_boxes),
                    "output_boxes": 0,
                    "status": "error",
                    "message": str(exc),
                })
                _set_progress_postfix(progress, status="error", crops=crop_count)
    finally:
        manifest_file.close()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--classes", required=True, help="classes.txt path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="VLLM model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None, help="API key; use EMPTY for local VLLM")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--task-type", default="", help="Optional task context, e.g. damaged traffic sign detection")
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.5, help="Class-wise NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=0, help="Limit images for smoke tests")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--no-conf", action="store_true", help="Write YOLO txt without confidence column")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen/VLLM post-check to YOLO txt")
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full-image", help="Detect defects from images and write YOLO txt")
    add_common_args(full)
    full.set_defaults(func=run_full_image)

    crop = sub.add_parser("crop-refine", help="Refine existing YOLO predictions using cropped candidates")
    add_common_args(crop)
    crop.add_argument("--pred-labels", required=True, help="Existing YOLO prediction txt directory")
    crop.add_argument("--crop-padding", type=float, default=0.15)
    crop.add_argument("--mode", choices=["classification", "detect"], default="classification")
    crop.set_defaults(func=run_crop_refine)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
