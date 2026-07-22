"""Smoke-test a Qwen/VLLM OpenAI-compatible vision endpoint with one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from classes import load_classes
from image_io import image_size
from parser import parse_detections
from prompts import full_image_prompt
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
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:18001/v1")
    parser.add_argument("--model", default=None, help="Model name exposed by VLLM")
    parser.add_argument("--api-key", default=None, help="API key; use EMPTY for local VLLM")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--raw-output", default="", help="Optional path to save raw model response")
    return parser.parse_args()


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
        prompt = full_image_prompt(classes)
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
        detections = parse_detections(response_text, width, height, require_bbox=False)
        print("=== parsed detections ===")
        print(json.dumps([
            {
                "class_name": det.class_name,
                "bbox_xyxy": list(det.xyxy) if det.xyxy is not None else None,
                "confidence": det.confidence,
                "keep": det.keep,
            }
            for det in detections
        ], ensure_ascii=False, indent=2))

    print("=== done ===")


if __name__ == "__main__":
    main()
