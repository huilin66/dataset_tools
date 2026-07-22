"""Prompts for Qwen/VLLM visual post-check."""

from __future__ import annotations


def class_list_text(classes: list[str]) -> str:
    return "\n".join(f"{idx}: {name}" for idx, name in enumerate(classes))


def full_image_prompt(classes: list[str]) -> str:
    return f"""
You are checking building facade defect detections.
Detect only objects whose class is in this class list:
{class_list_text(classes)}

Return strict JSON only, with no markdown:
{{
  "detections": [
    {{"class_name": "...", "bbox": [x1, y1, x2, y2], "confidence": 0.0, "keep": true}}
  ]
}}

bbox must be pixel coordinates in the image. If there is no valid defect, return {{"detections": []}}.
""".strip()


def crop_refine_prompt(classes: list[str], candidate_class: str) -> str:
    return f"""
You are verifying one candidate building facade defect crop.
The original YOLO candidate class is: {candidate_class}
Allowed classes:
{class_list_text(classes)}

Decide whether the candidate should be kept. If kept, correct the class if needed.
Return strict JSON only, with no markdown:
{{
  "detections": [
    {{"class_name": "...", "bbox": [x1, y1, x2, y2], "confidence": 0.0, "keep": true}}
  ]
}}

For classification-only refinement, bbox may cover the visible defect region in this crop. If the candidate is not a valid defect, return {{"detections": [{{"keep": false, "confidence": 0.0}}]}}.
""".strip()
