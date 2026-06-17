# -*- coding: utf-8 -*-
"""
使用 Ultralytics SAM3 直接分割图像中的所有 traffic sign。

输出：
output_dir/
├── overlay.jpg
├── combined_mask.png
├── result.json
├── labels/
│   └── image_name.txt
└── masks/
    ├── traffic_sign_001.png
    ├── traffic_sign_002.png
    └── ...

运行示例：
python sam3_all_traffic_signs.py \
    --image E:/data/traffic.jpg \
    --model E:/models/sam3.pt \
    --output-dir E:/results/sam3_traffic_sign
"""

import os

# 必须放在 torch 和 ultralytics 导入之前
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch._dynamo


# ============================================================
# 兼容部分 Ultralytics 版本的 torch.compile 参数问题
# ============================================================
torch._dynamo.config.disable = True

_original_torch_compile = torch.compile


def _safe_compile(model=None, *args, **kwargs):
    if kwargs.get("mode") is False:
        kwargs["mode"] = "default"

    kwargs["disable"] = True

    return _original_torch_compile(
        model,
        *args,
        **kwargs,
    )


torch.compile = _safe_compile


# 必须在 monkey patch 之后导入
from ultralytics.models.sam.predict import SAM3SemanticPredictor


def resize_binary_mask(
    mask,
    image_height: int,
    image_width: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """将 SAM3 mask 转为原图尺寸的 bool mask。"""

    if isinstance(mask, torch.Tensor):
        mask = (
            mask.detach()
            .float()
            .cpu()
            .numpy()
        )

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"无效 mask shape：{mask.shape}"
        )

    if mask.shape != (image_height, image_width):
        mask = cv2.resize(
            mask.astype(np.float32),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )

    if mask.max() > 1.0:
        return mask > 127

    return mask > threshold


def mask_to_bbox(mask: np.ndarray):
    """根据二值 mask 计算 xyxy bbox。"""

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def mask_to_largest_polygon(
    mask: np.ndarray,
    min_area: float = 20.0,
    simplify_ratio: float = 0.001,
):
    """
    将二值 mask 转换为最大外轮廓 polygon。

    返回：
        numpy.ndarray，shape=[N, 2]
    """

    binary = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
        and len(contour) >= 3
    ]

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    perimeter = cv2.arcLength(
        contour,
        closed=True,
    )

    epsilon = max(
        0.5,
        simplify_ratio * perimeter,
    )

    polygon = cv2.approxPolyDP(
        contour,
        epsilon,
        closed=True,
    )

    polygon = (
        polygon.reshape(-1, 2)
        .astype(np.float32)
    )

    if len(polygon) < 3:
        return None

    return polygon


def polygon_to_yolo_line(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    class_id: int = 0,
):
    """
    转换为 YOLO instance-seg 标签行：

    class_id x1 y1 x2 y2 ... xn yn
    """

    polygon = polygon.astype(np.float32).copy()

    polygon[:, 0] /= float(image_width)
    polygon[:, 1] /= float(image_height)

    polygon = np.clip(
        polygon,
        0.0,
        1.0,
    )

    coordinates = " ".join(
        f"{value:.6f}"
        for point in polygon
        for value in point
    )

    return f"{class_id} {coordinates}"


def segment_all_traffic_signs(
    image_path,
    model_path,
    output_dir,
    prompt="traffic sign panel",
    conf=0.25,
    mask_threshold=0.5,
    min_mask_area=30,
    device="cuda:0",
):
    """
    使用 SAM3 文本提示分割图像中的所有交通标志。

    Args:
        image_path:
            输入图像路径。

        model_path:
            sam3.pt 路径。

        output_dir:
            输出目录。

        prompt:
            SAM3 文本提示，默认 traffic sign。

        conf:
            SAM3 实例置信度阈值。

        mask_threshold:
            mask 二值化阈值。

        min_mask_area:
            小于该像素面积的 mask 将被过滤。

        device:
            cuda:0、cuda:1 或 cpu。

    Returns:
        包含所有交通标志实例信息的字典。
    """

    image_path = Path(
        image_path
    ).expanduser().resolve()

    model_path = Path(
        model_path
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"图像不存在：{image_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"SAM3 模型不存在：{model_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dir = output_dir / "masks"
    label_dir = output_dir / "labels"

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV 无法读取图像：{image_path}"
        )

    image_height, image_width = image.shape[:2]

    overrides = {
        "conf": conf,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "device": device,
        "half": device != "cpu",
        "compile": False,
        "save": False,
        "show": False,
        "verbose": False,
    }

    predictor = None

    try:
        print("====== 正在加载 SAM3 ======")

        predictor = SAM3SemanticPredictor(
            overrides=overrides
        )

        print("====== 正在提取图像特征 ======")

        # 同一图像只提取一次特征
        predictor.set_image(image)

        print(
            f"====== 使用文本提示分割："
            f"{prompt!r} ======"
        )

        # SAM3 Semantic Predictor 会查找所有匹配实例
        results = predictor(
            text=[prompt],
            stream=False,
            save=False,
            show=False,
            verbose=False,
        )

        if not isinstance(results, list):
            results = list(results)

        if not results:
            raise RuntimeError(
                "SAM3 没有返回 Results。"
            )

        result = results[0]

        if (
            result.masks is None
            or result.masks.data is None
            or result.masks.data.numel() == 0
        ):
            print("未检测到交通标志。")

            empty_result = {
                "image": str(image_path),
                "prompt": prompt,
                "image_width": image_width,
                "image_height": image_height,
                "num_objects": 0,
                "objects": [],
            }

            result_json_path = (
                output_dir / "result.json"
            )

            with result_json_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    empty_result,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            # 创建空标签文件
            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )
            label_path.touch()

            return empty_result

        masks_tensor = result.masks.data

        # 获取置信度
        if (
            result.boxes is not None
            and result.boxes.conf is not None
        ):
            scores = (
                result.boxes.conf.detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            scores = np.ones(
                len(masks_tensor),
                dtype=np.float32,
            )

        # 获取 SAM3 bbox
        if (
            result.boxes is not None
            and result.boxes.xyxy is not None
        ):
            predicted_boxes = (
                result.boxes.xyxy.detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            predicted_boxes = None

        overlay = image.copy()

        combined_mask = np.zeros(
            (image_height, image_width),
            dtype=bool,
        )

        objects = []
        yolo_lines = []

        valid_index = 0

        for raw_index, mask_tensor in enumerate(
            masks_tensor
        ):
            mask = resize_binary_mask(
                mask=mask_tensor,
                image_height=image_height,
                image_width=image_width,
                threshold=mask_threshold,
            )

            mask_area = int(mask.sum())

            if mask_area < min_mask_area:
                print(
                    f"[Filter] mask {raw_index} 面积过小："
                    f"{mask_area}"
                )
                continue

            score = (
                float(scores[raw_index])
                if raw_index < len(scores)
                else 1.0
            )

            if score < conf:
                continue

            valid_index += 1

            # 优先采用 SAM3 返回的 bbox
            if (
                predicted_boxes is not None
                and raw_index < len(predicted_boxes)
            ):
                bbox = [
                    float(value)
                    for value
                    in predicted_boxes[raw_index]
                ]
            else:
                bbox = mask_to_bbox(mask)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            x1 = max(
                0,
                min(image_width - 1, int(round(x1))),
            )
            y1 = max(
                0,
                min(image_height - 1, int(round(y1))),
            )
            x2 = max(
                0,
                min(image_width - 1, int(round(x2))),
            )
            y2 = max(
                0,
                min(image_height - 1, int(round(y2))),
            )

            bbox = [x1, y1, x2, y2]

            combined_mask |= mask

            # 保存单实例 mask
            mask_filename = (
                f"traffic_sign_{valid_index:03d}.png"
            )

            mask_path = mask_dir / mask_filename

            cv2.imwrite(
                str(mask_path),
                mask.astype(np.uint8) * 255,
            )

            # 转换 YOLO polygon
            polygon = mask_to_largest_polygon(
                mask=mask,
                min_area=min_mask_area,
                simplify_ratio=0.001,
            )

            polygon_list = []

            if polygon is not None:
                polygon_list = (
                    polygon.astype(int).tolist()
                )

                yolo_line = polygon_to_yolo_line(
                    polygon=polygon,
                    image_width=image_width,
                    image_height=image_height,
                    class_id=0,
                )

                yolo_lines.append(yolo_line)

            # 稳定可复现的颜色
            color = (
                int((37 * valid_index) % 255),
                int((97 * valid_index) % 255),
                int((173 * valid_index) % 255),
            )

            # mask 半透明叠加
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.50
                + np.asarray(
                    color,
                    dtype=np.float32,
                ) * 0.50
            ).astype(np.uint8)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            label = (
                f"traffic_sign {score:.3f}"
            )

            cv2.putText(
                overlay,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            bbox_norm_1000 = [
                round(x1 / image_width * 1000),
                round(y1 / image_height * 1000),
                round(x2 / image_width * 1000),
                round(y2 / image_height * 1000),
            ]

            objects.append(
                {
                    "id": valid_index,
                    "class_id": 0,
                    "class_name": "traffic_sign",
                    "score": round(score, 6),
                    "bbox_xyxy": bbox,
                    "bbox_norm_1000": bbox_norm_1000,
                    "mask_area_pixels": mask_area,
                    "mask_area_ratio": round(
                        mask_area
                        / float(
                            image_width * image_height
                        ),
                        8,
                    ),
                    "mask_path": str(mask_path),
                    "polygon": polygon_list,
                }
            )

            print(
                f"[{valid_index}] "
                f"score={score:.4f}, "
                f"bbox={bbox}, "
                f"area={mask_area}"
            )

        # 保存所有实例合并 mask
        combined_mask_path = (
            output_dir / "combined_mask.png"
        )

        cv2.imwrite(
            str(combined_mask_path),
            combined_mask.astype(np.uint8) * 255,
        )

        # 保存可视化
        overlay_path = output_dir / "overlay.jpg"

        cv2.imwrite(
            str(overlay_path),
            overlay,
        )

        # 保存 YOLO instance segmentation TXT
        label_path = (
            label_dir / f"{image_path.stem}.txt"
        )

        with label_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            if yolo_lines:
                file.write("\n".join(yolo_lines))
                file.write("\n")

        final_result = {
            "image": str(image_path),
            "prompt": prompt,
            "image_width": image_width,
            "image_height": image_height,
            "num_objects": len(objects),
            "objects": objects,
            "overlay_path": str(overlay_path),
            "combined_mask_path": str(
                combined_mask_path
            ),
            "yolo_label_path": str(label_path),
            "yolo_names": {
                "0": "traffic_sign"
            },
        }

        result_json_path = (
            output_dir / "result.json"
        )

        with result_json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                final_result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("=" * 60)
        print(
            f"检测到交通标志数量："
            f"{len(objects)}"
        )
        print(f"JSON：{result_json_path}")
        print(f"可视化：{overlay_path}")
        print(f"合并 mask：{combined_mask_path}")
        print(f"YOLO TXT：{label_path}")
        print(f"单实例 mask：{mask_dir}")
        print("=" * 60)

        return final_result

    finally:
        if predictor is not None:
            try:
                predictor.reset_image()
            except Exception:
                pass

            del predictor

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "使用 Ultralytics SAM3 "
            "分割图像中的所有交通标志"
        )
    )

    parser.add_argument(
        "--image",
        default=r'\\158.132.186.40\isds\huilin\traffic_sign\demo_0617\images\DA5324655_20250806125609500.jpg',
        help="输入图像路径",
    )

    parser.add_argument(
        "--model",
        default=r"llm_tools/vllm/sam3.pt",
        help="sam3.pt 路径",
    )

    parser.add_argument(
        "--output-dir",
        default="./sam3_traffic_sign_results",
        help="结果输出目录",
    )

    parser.add_argument(
        "--prompt",
        default="traffic sign face",
        help="SAM3 文本概念提示",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="SAM3 置信度阈值",
    )

    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=30,
        help="最小 mask 像素面积",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="例如 cuda:0、cuda:1 或 cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    segment_all_traffic_signs(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.output_dir,
        prompt=args.prompt,
        conf=args.conf,
        mask_threshold=args.mask_threshold,
        min_mask_area=args.min_mask_area,
        device=args.device,
    )


if __name__ == "__main__":
    main()