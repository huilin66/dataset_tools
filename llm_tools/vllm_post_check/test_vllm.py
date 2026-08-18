"""Smoke-test a Qwen/VLLM OpenAI-compatible vision endpoint with one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classes import load_classes
from image_io import image_size
from parser import parse_detections, parse_p2_full_detections
from prompts import PROMPT_VERSIONS, full_image_prompt, p2_full_image_prompt
from vllm_client import VLLMClient


SIMPLE_PROMPT = """
You are testing whether a vision-language model can read this image.
Return strict JSON only, with no markdown:
{
  "ok": true,
  "image_visible": true,
  "summary": "one short sentence about the image"
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen/VLLM vision prediction with one image")
    parser.add_argument("--image", required=True, help="One image path")
    parser.add_argument("--classes", default="", help="Optional classes.txt. Required for --mode detect.")
    parser.add_argument("--mode", choices=["simple", "detect"], default="simple")
    parser.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default="p1",
        help="Detection prompt: p1 generic or p2 Qwen3-VL-SFT-compatible",
    )
    parser.add_argument(
        "--p2-confidence",
        type=float,
        default=1.0,
        help="Fixed confidence shown for parsed p2 detections",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:18001/v1")
    parser.add_argument("--model", default=None, help="Model name exposed by VLLM")
    parser.add_argument("--api-key", default=None, help="API key; use EMPTY for local VLLM")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--task-type", default="", help="Optional task context for --mode detect")
    parser.add_argument("--raw-output", default="", help="Optional path to save raw model response")
    args = parser.parse_args()
    if not 0.0 <= args.p2_confidence <= 1.0:
        parser.error("--p2-confidence must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    width, height = image_size(image_path)
    if args.mode == "detect":
        if not args.classes:
            raise ValueError("--classes is required when --mode detect")
        classes = load_classes(args.classes)
        prompt = (
            p2_full_image_prompt(classes, args.task_type)
            if args.prompt_version == "p2"
            else full_image_prompt(classes, args.task_type)
        )
    else:
        classes = []
        prompt = SIMPLE_PROMPT

    client = VLLMClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("=== VLLM smoke test ===")
    print(f"image: {image_path}")
    print(f"image_size: {width}x{height}")
    print(f"base_url: {client.base_url}")
    print(f"model: {client.model}")
    print(f"mode: {args.mode}")
    print("sending request...")

    response_text = client.predict(image_path, prompt)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(response_text, encoding="utf-8")
        print(f"raw saved: {raw_path}")

    print("=== raw response ===")
    print(response_text)

    if args.mode == "detect":
        if args.prompt_version == "p2":
            detections = parse_p2_full_detections(
                response_text, args.p2_confidence
            )
            parsed = [
                {
                    "class_name": det.class_name,
                    "bbox_norm_1000": list(det.bbox_norm_1000),
                    "confidence": det.confidence,
                    "keep": True,
                }
                for det in detections
            ]
        else:
            detections = parse_detections(
                response_text, width, height, require_bbox=False
            )
            parsed = [
                {
                    "class_name": det.class_name,
                    "bbox_xyxy": (
                        list(det.xyxy) if det.xyxy is not None else None
                    ),
                    "confidence": det.confidence,
                    "keep": det.keep,
                }
                for det in detections
            ]
        print("=== parsed detections ===")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))

    print("=== done ===")


if __name__ == "__main__":
    main()

