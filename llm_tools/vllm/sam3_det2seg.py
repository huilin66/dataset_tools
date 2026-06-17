# -*- coding: utf-8 -*-
"""
Qwen3-VL bbox -> Ultralytics SAM3 mask

输入：
1. 一张原始图像
2. Qwen3-VL 输出的 JSON

输出：
1. 每个交通标志的二值 mask
2. mask 叠加可视化图
3. 合并类型、bbox、mask、polygon 的 JSON

运行示例：
python sam3_qwen_box.py \
    --image E:/data/traffic.jpg \
    --qwen-json E:/data/traffic_sign_result.json \
    --model E:/models/sam3.pt \
    --output-dir E:/data/sam3_results
"""

import os

# 必须在导入 torch 和 ultralytics 前设置
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch._dynamo
from ultralytics.engine.results import Results

# ============================================================
# 修复部分 Ultralytics 版本把 False 传给 torch.compile(mode)
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


# 必须在 monkey patch 后导入
from ultralytics.models.sam.predict import SAM3Predictor


class TrafficSignSAM3Predictor(SAM3Predictor):
    """
    基于 Ultralytics SAM3Predictor 的交通标志 bbox 分割器。

    功能：
    1. 图像只编码一次；
    2. 接收 Qwen3-VL 的 bbox；
    3. 逐框调用 SAM3 mask decoder；
    4. 保存二值 mask、polygon 和叠加图；
    5. 保留 Qwen 输出的交通标志类型。
    """

    def __init__(self, overrides=None):
        overrides = dict(overrides or {})

        overrides.setdefault("task", "segment")
        overrides.setdefault("mode", "predict")
        overrides.setdefault("compile", False)
        overrides.setdefault("save", False)
        overrides.setdefault("show", False)
        overrides.setdefault("verbose", False)

        super().__init__(overrides=overrides)

        self.original_image: np.ndarray | None = None
        self.image_height: int | None = None
        self.image_width: int | None = None

    def set_traffic_image(
        self,
        image,
    ):
        """
        设置图像并提取一次图像特征。

        后续所有 bbox 都复用该图像特征。
        """
        if isinstance(image, (str, os.PathLike)):
            image_path = Path(image).expanduser().resolve()

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"图像不存在：{image_path}"
                )

            cv_image = cv2.imread(str(image_path))

            if cv_image is None:
                raise ValueError(
                    f"OpenCV 无法读取图像：{image_path}"
                )

        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(
                    f"图像维度必须为 HWC，当前为：{image.shape}"
                )

            cv_image = image.copy()

        else:
            raise TypeError(
                "image 必须是图像路径或 numpy.ndarray。"
            )

        self.original_image = cv_image
        self.image_height, self.image_width = cv_image.shape[:2]

        print(
            f"设置图像："
            f"{self.image_width} × {self.image_height}"
        )

        # Ultralytics 会提取并缓存图像特征
        super().set_image(cv_image)

        return self

    @staticmethod
    def _safe_filename(value: Any) -> str:
        value = str(value or "unknown")
        value = re.sub(
            r"[^0-9a-zA-Z_\-]+",
            "_",
            value,
        )
        return value.strip("_") or "unknown"

    def _clip_and_expand_box(
        self,
        bbox_xyxy,
        expand_ratio,
    ):
        """
        裁剪 bbox 到图像范围，并向四周轻微扩张。
        """
        if self.image_width is None or self.image_height is None:
            raise RuntimeError("请先调用 set_traffic_image()。")

        if len(bbox_xyxy) != 4:
            raise ValueError(
                f"bbox 必须包含4个数值：{bbox_xyxy}"
            )

        x1, y1, x2, y2 = map(float, bbox_xyxy)

        # 兼容错误的坐标顺序
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"无效 bbox：{bbox_xyxy}"
            )

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * expand_ratio
        pad_y = box_height * expand_ratio

        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)

        x2 = min(
            float(self.image_width - 1),
            x2 + pad_x,
        )
        y2 = min(
            float(self.image_height - 1),
            y2 + pad_y,
        )

        return [x1, y1, x2, y2]

    def qwen_bbox_to_xyxy(
        self,
        obj: dict,
        expand_ratio = 0.02,
    ):
        """
        将 Qwen 输出框转换为原图像素 xyxy。

        支持字段：
        1. bbox_norm_1000：0~1000 归一化坐标
        2. bbox_xyxy：原图像素坐标
        3. bbox：原图像素坐标
        """
        if self.image_width is None or self.image_height is None:
            raise RuntimeError("请先调用 set_traffic_image()。")

        if "bbox_norm_1000" in obj:
            nx1, ny1, nx2, ny2 = map(
                float,
                obj["bbox_norm_1000"],
            )

            bbox_xyxy = [
                nx1 / 1000.0 * self.image_width,
                ny1 / 1000.0 * self.image_height,
                nx2 / 1000.0 * self.image_width,
                ny2 / 1000.0 * self.image_height,
            ]

        elif "bbox_xyxy" in obj:
            bbox_xyxy = list(
                map(float, obj["bbox_xyxy"])
            )

        elif "bbox" in obj:
            bbox_xyxy = list(
                map(float, obj["bbox"])
            )

        else:
            raise KeyError(
                "Qwen 对象中缺少 bbox_norm_1000、"
                "bbox_xyxy 或 bbox。"
            )

        return self._clip_and_expand_box(
            bbox_xyxy=bbox_xyxy,
            expand_ratio=expand_ratio,
        )

    @staticmethod
    def mask_to_polygon(
        mask: np.ndarray,
        simplify_ratio = 0.002,
    ):
        """
        将 mask 转换为最大外轮廓 polygon。
        """
        binary_mask = (
            mask > 0
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        contour = max(
            contours,
            key=cv2.contourArea,
        )

        if len(contour) < 3:
            return []

        perimeter = cv2.arcLength(
            contour,
            closed=True,
        )

        epsilon = max(
            1.0,
            simplify_ratio * perimeter,
        )

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        return (
            contour
            .reshape(-1, 2)
            .astype(int)
            .tolist()
        )

    @staticmethod
    def mask_to_bbox(
        mask,
    ):
        """
        根据 mask 计算 xyxy bbox。
        """
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    @staticmethod
    def _get_result_score(result):
        """
        从 Ultralytics Results 中安全读取分数。
        """
        try:
            if (
                result.boxes is not None
                and result.boxes.conf is not None
                and len(result.boxes.conf) > 0
            ):
                return float(
                    result.boxes.conf[0]
                    .detach()
                    .cpu()
                )
        except (AttributeError, IndexError, TypeError):
            pass

        return None

    def predict_single_box_mask(
        self,
        bbox_xyxy,
        mask_threshold = 0.5,
    ):
        """
        使用一个 bbox 对一个对象执行 mask 推理。

        因为已经调用 set_traffic_image()，这里不会重新运行
        图像 encoder，只运行 prompt encoder 和 mask decoder。
        """
        if self.original_image is None:
            raise RuntimeError(
                "请先调用 set_traffic_image()。"
            )

        # 防止上一次提示残留
        if hasattr(self, "prompts"):
            self.prompts = {}

        results = self(
            bboxes=[bbox_xyxy],
            multimask_output=False,
            stream=False,
            save=False,
            verbose=False,
        )

        if not isinstance(results, list):
            results = list(results)

        if len(results) == 0:
            raise RuntimeError(
                f"SAM3 未返回结果，bbox={bbox_xyxy}"
            )

        result = results[0]

        if result.masks is None:
            raise RuntimeError(
                f"SAM3 未生成 mask，bbox={bbox_xyxy}"
            )

        mask_tensor = result.masks.data

        if mask_tensor is None or mask_tensor.numel() == 0:
            raise RuntimeError(
                f"SAM3 mask 为空，bbox={bbox_xyxy}"
            )

        # 单框、multimask_output=False，取第一个 mask
        mask = (
            mask_tensor[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        mask = mask > mask_threshold

        # 某些版本返回的是推理尺寸，需要还原到原图大小
        if mask.shape != (
            self.image_height,
            self.image_width,
        ):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (
                    self.image_width,
                    self.image_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        score = self._get_result_score(result)

        return mask, score

    def segment_qwen_objects(
        self,
        qwen_result,
        output_dir,
        expand_ratio = 0.02,
        mask_threshold = 0.5,
        overlay_alpha = 0.45,
    ):
        """
        读取 Qwen 检测结果并逐框生成 SAM3 mask。
        """
        if self.original_image is None:
            raise RuntimeError(
                "请先调用 set_traffic_image()。"
            )

        # ------------------------------------------------
        # 读取 Qwen JSON
        # ------------------------------------------------
        if isinstance(qwen_result, dict):
            qwen_data = qwen_result

        elif isinstance(
            qwen_result,
            (str, os.PathLike),
        ):
            json_path = Path(
                qwen_result
            ).expanduser().resolve()

            if not json_path.is_file():
                raise FileNotFoundError(
                    f"Qwen JSON 不存在：{json_path}"
                )

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                qwen_data = json.load(file)

        else:
            raise TypeError(
                "qwen_result 必须是 dict 或 JSON 文件路径。"
            )

        objects = qwen_data.get("objects", [])

        if not isinstance(objects, list):
            raise TypeError(
                "Qwen JSON 中 objects 必须是列表。"
            )

        output_dir = Path(
            output_dir
        ).expanduser().resolve()

        mask_dir = output_dir / "masks"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        overlay = self.original_image.copy()
        final_objects = []

        # ------------------------------------------------
        # 每个 Qwen bbox 单独分割
        # ------------------------------------------------
        for index, obj in enumerate(objects, start=1):
            object_id = obj.get("id", index)

            specific_type = obj.get(
                "specific_type",
                obj.get("category", "unknown"),
            )

            chinese_name = obj.get(
                "chinese_name",
                "",
            )

            try:
                bbox_xyxy = self.qwen_bbox_to_xyxy(
                    obj=obj,
                    expand_ratio=expand_ratio,
                )

                print(
                    f"[{index}/{len(objects)}] "
                    f"id={object_id}, "
                    f"type={specific_type}, "
                    f"bbox={bbox_xyxy}"
                )

                mask, sam3_score = (
                    self.predict_single_box_mask(
                        bbox_xyxy=bbox_xyxy,
                        mask_threshold=mask_threshold,
                    )
                )

            except Exception as exc:
                print(
                    f"[Warning] 对象 {object_id} 分割失败：{exc}"
                )

                failed_obj = dict(obj)

                failed_obj.update(
                    {
                        "sam3_success": False,
                        "sam3_error": str(exc),
                    }
                )

                final_objects.append(failed_obj)
                continue

            mask_uint8 = (
                mask.astype(np.uint8) * 255
            )

            filename_type = self._safe_filename(
                specific_type
            )

            mask_filename = (
                f"{index:03d}_{filename_type}.png"
            )

            mask_path = mask_dir / mask_filename

            if not cv2.imwrite(
                str(mask_path),
                mask_uint8,
            ):
                raise IOError(
                    f"无法保存 mask：{mask_path}"
                )

            polygon = self.mask_to_polygon(
                mask_uint8
            )

            mask_bbox_xyxy = self.mask_to_bbox(
                mask_uint8
            )

            mask_area = int(mask.sum())

            # ------------------------------------------------
            # 可视化
            # ------------------------------------------------
            color = np.array(
                [
                    (37 * index) % 255,
                    (97 * index) % 255,
                    (173 * index) % 255,
                ],
                dtype=np.float32,
            )

            mask_region = mask.astype(bool)

            overlay[mask_region] = (
                overlay[mask_region].astype(np.float32)
                * (1.0 - overlay_alpha)
                + color * overlay_alpha
            ).astype(np.uint8)

            x1, y1, x2, y2 = map(
                int,
                bbox_xyxy,
            )

            draw_color = tuple(
                int(value)
                for value in color.tolist()
            )

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                draw_color,
                2,
            )

            # OpenCV 默认字体不支持中文，因此显示英文类型
            label_text = (
                f"{object_id}: {specific_type}"
            )

            if sam3_score is not None:
                label_text += f" {sam3_score:.3f}"

            cv2.putText(
                overlay,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                draw_color,
                2,
                cv2.LINE_AA,
            )

            final_obj = dict(obj)

            final_obj.update(
                {
                    "sam3_success": True,
                    "bbox_xyxy": [
                        round(float(value), 2)
                        for value in bbox_xyxy
                    ],
                    "sam3_mask_bbox_xyxy": mask_bbox_xyxy,
                    "sam3_score": (
                        round(sam3_score, 6)
                        if sam3_score is not None
                        else None
                    ),
                    "mask_area_pixels": mask_area,
                    "mask_path": str(mask_path),
                    "polygon": polygon,
                    "specific_type": specific_type,
                    "chinese_name": chinese_name,
                }
            )

            final_objects.append(final_obj)

        # ------------------------------------------------
        # 保存结果
        # ------------------------------------------------
        overlay_path = output_dir / "overlay.jpg"

        if not cv2.imwrite(
            str(overlay_path),
            overlay,
        ):
            raise IOError(
                f"无法保存叠加图：{overlay_path}"
            )

        final_result = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "num_input_objects": len(objects),
            "num_success_objects": sum(
                bool(obj.get("sam3_success"))
                for obj in final_objects
            ),
            "objects": final_objects,
            "overlay_path": str(overlay_path),
        }

        final_json_path = (
            output_dir / "sam3_result.json"
        )

        with final_json_path.open(
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
        print("SAM3 bbox 分割完成")
        print(f"JSON:   {final_json_path}")
        print(f"Overlay:{overlay_path}")
        print(f"Masks:  {mask_dir}")
        print("=" * 60)

        return final_result
    def sam_to_yolo_instance_seg_result(
        self,
        orig_img,
        sam_output,
        image_path="",
        names=None,
        class_key="specific_type",
        confidence_key="confidence",
        mask_threshold=0.5,
        device="cpu",
    ):
        """
        将 SAM3 + Qwen 输出转换成 Ultralytics YOLO Instance Segmentation Results。

        Args:
            orig_img:
                原始 OpenCV BGR 图像，shape=(H, W, 3)。

            sam_output:
                可以是：
                1. segment_qwen_objects() 返回的字典；
                2. sam3_result.json 文件路径。

                其中 objects 中每个对象应包含：
                {
                    "specific_type": "speed_limit_50",
                    "confidence": 0.95,
                    "sam3_score": 0.92,
                    "mask_path": ".../001_speed_limit_50.png",
                    "bbox_xyxy": [x1, y1, x2, y2]
                }

            image_path:
                原始图片路径，仅用于 Ultralytics Results.path。

            names:
                可选的固定 YOLO 类别映射，例如：
                {
                    0: "speed_limit",
                    1: "stop",
                    2: "no_parking"
                }

                如果不提供，会根据 specific_type 自动建立类别映射。

            class_key:
                从对象中读取类别名称的字段，默认 specific_type。

            confidence_key:
                优先作为 YOLO confidence 的字段，默认 confidence。
                如果不存在，则依次使用 sam3_score 和 1.0。

            mask_threshold:
                mask 二值化阈值。

            device:
                Results tensor 所在设备。一般使用 cpu 即可进行可视化。

        Returns:
            ultralytics.engine.results.Results
        """
        if orig_img is None:
            raise ValueError("orig_img 不能为空。")

        if not isinstance(orig_img, np.ndarray):
            raise TypeError(
                f"orig_img 必须是 numpy.ndarray，当前为 {type(orig_img)}"
            )

        if orig_img.ndim != 3:
            raise ValueError(
                f"orig_img 必须为 HWC 图像，当前 shape={orig_img.shape}"
            )

        image_height, image_width = orig_img.shape[:2]

        # --------------------------------------------------
        # 1. 读取 SAM 输出
        # --------------------------------------------------
        json_base_dir = Path.cwd()

        if isinstance(sam_output, (str, os.PathLike, Path)):
            sam_json_path = Path(
                sam_output
            ).expanduser().resolve()

            if not sam_json_path.is_file():
                raise FileNotFoundError(
                    f"SAM JSON 文件不存在：{sam_json_path}"
                )

            json_base_dir = sam_json_path.parent

            with sam_json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                sam_data = json.load(file)

        elif isinstance(sam_output, dict):
            sam_data = sam_output

        else:
            raise TypeError(
                "sam_output 必须是 dict 或 JSON 文件路径。"
            )

        objects = sam_data.get("objects", [])

        if not isinstance(objects, list):
            raise TypeError("sam_output['objects'] 必须是列表。")

        # --------------------------------------------------
        # 2. 构造类别映射
        # --------------------------------------------------
        if names is not None:
            yolo_names = {
                int(class_id): str(class_name)
                for class_id, class_name in names.items()
            }

            name_to_id = {
                class_name: class_id
                for class_id, class_name in yolo_names.items()
            }

        else:
            yolo_names = {}
            name_to_id = {}

        boxes_data = []
        masks_data = []

        # --------------------------------------------------
        # 3. 遍历每个 SAM 实例
        # --------------------------------------------------
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                print(
                    f"[Warning] 跳过第 {object_index} 个对象："
                    "对象不是字典。"
                )
                continue

            if obj.get("sam3_success") is False:
                continue

            # ----------------------------------------------
            # 3.1 类别
            # ----------------------------------------------
            class_name = str(
                obj.get(
                    class_key,
                    obj.get(
                        "category",
                        "unknown",
                    ),
                )
            )

            if "class_id" in obj:
                class_id = int(obj["class_id"])

                if class_id not in yolo_names:
                    yolo_names[class_id] = class_name

                name_to_id[class_name] = class_id

            elif class_name in name_to_id:
                class_id = name_to_id[class_name]

            else:
                if names is not None:
                    print(
                        f"[Warning] 类别 {class_name!r} "
                        "不在固定 names 中，跳过该对象。"
                    )
                    continue

                class_id = len(yolo_names)
                yolo_names[class_id] = class_name
                name_to_id[class_name] = class_id

            # ----------------------------------------------
            # 3.2 读取 mask
            # ----------------------------------------------
            mask = None

            # 支持直接传入内存 mask
            if "mask" in obj and obj["mask"] is not None:
                mask_value = obj["mask"]

                if isinstance(mask_value, torch.Tensor):
                    mask = (
                        mask_value
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )

                elif isinstance(mask_value, np.ndarray):
                    mask = mask_value.copy()

                else:
                    mask = np.asarray(mask_value)

            # 支持读取保存后的 mask 文件
            elif obj.get("mask_path"):
                mask_path = Path(
                    obj["mask_path"]
                ).expanduser()

                if not mask_path.is_absolute():
                    mask_path = (
                        json_base_dir / mask_path
                    ).resolve()

                if not mask_path.is_file():
                    print(
                        f"[Warning] mask 文件不存在，跳过："
                        f"{mask_path}"
                    )
                    continue

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

            if mask is None:
                print(
                    f"[Warning] 对象 {object_index} "
                    "没有可用 mask，跳过。"
                )
                continue

            # 去除多余维度
            mask = np.squeeze(mask)

            if mask.ndim != 2:
                print(
                    f"[Warning] 对象 {object_index} "
                    f"mask shape 无效：{mask.shape}"
                )
                continue

            # 恢复至原图大小
            if mask.shape != (
                image_height,
                image_width,
            ):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 同时兼容：
            # 0/1 mask
            # 0/255 mask
            # float logits/probability mask
            if mask.max() > 1.0:
                binary_mask = mask > 127
            else:
                binary_mask = mask > mask_threshold

            if not binary_mask.any():
                print(
                    f"[Warning] 对象 {object_index} "
                    "mask 为空，跳过。"
                )
                continue

            # ----------------------------------------------
            # 3.3 bbox
            # ----------------------------------------------
            bbox = None

            # 优先使用 mask 计算出的 bbox
            if obj.get("sam3_mask_bbox_xyxy") is not None:
                bbox = obj["sam3_mask_bbox_xyxy"]

            elif obj.get("bbox_xyxy") is not None:
                bbox = obj["bbox_xyxy"]

            elif obj.get("bbox") is not None:
                bbox = obj["bbox"]

            if bbox is not None:
                x1, y1, x2, y2 = map(float, bbox)

            else:
                # 根据 mask 自动计算 bbox
                ys, xs = np.where(binary_mask)

                x1 = float(xs.min())
                y1 = float(ys.min())
                x2 = float(xs.max())
                y2 = float(ys.max())

            # 处理坐标顺序
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            # 限制到图像范围
            x1 = np.clip(x1, 0, image_width - 1)
            y1 = np.clip(y1, 0, image_height - 1)
            x2 = np.clip(x2, 0, image_width - 1)
            y2 = np.clip(y2, 0, image_height - 1)

            # ----------------------------------------------
            # 3.4 confidence
            # ----------------------------------------------
            confidence = obj.get(confidence_key)

            if confidence is None:
                confidence = obj.get("sam3_score")

            if confidence is None:
                confidence = 1.0

            confidence = float(
                np.clip(
                    confidence,
                    0.0,
                    1.0,
                )
            )

            # Ultralytics boxes 格式：
            # [x1, y1, x2, y2, confidence, class_id]
            boxes_data.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    confidence,
                    float(class_id),
                ]
            )

            masks_data.append(
                torch.from_numpy(
                    binary_mask.astype(np.uint8)
                )
            )

        # --------------------------------------------------
        # 4. 构造 Ultralytics tensor
        # --------------------------------------------------
        if boxes_data:
            boxes_tensor = torch.tensor(
                boxes_data,
                dtype=torch.float32,
                device=device,
            )

            masks_tensor = torch.stack(
                masks_data,
                dim=0,
            ).to(
                device=device,
                dtype=torch.uint8,
            )

        else:
            boxes_tensor = torch.empty(
                (0, 6),
                dtype=torch.float32,
                device=device,
            )

            masks_tensor = torch.empty(
                (
                    0,
                    image_height,
                    image_width,
                ),
                dtype=torch.uint8,
                device=device,
            )

        # --------------------------------------------------
        # 5. 创建原生 Ultralytics Results
        # --------------------------------------------------
        yolo_result = Results(
            orig_img=orig_img,
            path=str(image_path),
            names=yolo_names,
            boxes=boxes_tensor,
            masks=masks_tensor,
        )

        return yolo_result
    
    def export_yolo_seg_txt(
        self,
        sam_result,
        txt_path,
        class_name_to_id=None,
        class_key="specific_type",
        min_area=20.0,
        simplify_ratio=0.001,
        use_largest_contour=True,
    ):
        """
        将 SAM3 分割结果转换为 YOLO Instance Segmentation TXT。

        YOLO segmentation 每一行格式：
            class_id x1 y1 x2 y2 ... xn yn

        坐标均归一化至 [0, 1]。

        Args:
            sam_result:
                segment_qwen_objects() 返回的字典，
                或 sam3_result.json 文件路径。

            txt_path:
                输出 TXT 文件路径。

            class_name_to_id:
                固定类别映射，例如：
                {
                    "speed_limit_50": 0,
                    "stop": 1,
                    "no_parking": 2
                }

                如果对象自身含有 class_id，则优先使用 class_id。

            class_key:
                从对象中读取类别名称的字段。
                默认使用 specific_type。

            min_area:
                忽略面积小于该值的轮廓，单位为像素。

            simplify_ratio:
                多边形简化比例。
                值越大，polygon 点数越少。

            use_largest_contour:
                True：
                    每个实例只保留最大轮廓，推荐用于交通标志。
                False：
                    一个 mask 中的每个外轮廓分别写成一个实例。

        Returns:
            dict:
                {
                    "txt_path": "...",
                    "num_instances": 3,
                    "num_skipped": 1
                }
        """
        if self.original_image is None:
            raise RuntimeError(
                "请先调用 set_traffic_image() 设置图像。"
            )

        image_height, image_width = self.original_image.shape[:2]

        # --------------------------------------------------
        # 1. 读取 SAM 结果
        # --------------------------------------------------
        json_base_dir = Path.cwd()

        if isinstance(sam_result, dict):
            sam_data = sam_result

        elif isinstance(sam_result, (str, os.PathLike, Path)):
            sam_json_path = Path(
                sam_result
            ).expanduser().resolve()

            if not sam_json_path.is_file():
                raise FileNotFoundError(
                    f"SAM JSON 不存在：{sam_json_path}"
                )

            json_base_dir = sam_json_path.parent

            with sam_json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                sam_data = json.load(file)

        else:
            raise TypeError(
                "sam_result 必须是 dict 或 JSON 文件路径。"
            )

        objects = sam_data.get("objects", [])

        if not isinstance(objects, list):
            raise TypeError(
                "sam_result['objects'] 必须是 list。"
            )

        txt_path = Path(
            txt_path
        ).expanduser().resolve()

        txt_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        class_name_to_id = class_name_to_id or {}

        output_lines = []
        skipped_count = 0

        # --------------------------------------------------
        # 2. 遍历每个 SAM 实例
        # --------------------------------------------------
        for object_index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                skipped_count += 1
                continue

            if obj.get("sam3_success") is False:
                skipped_count += 1
                continue

            # ----------------------------------------------
            # 2.1 获取类别 ID
            # ----------------------------------------------
            class_name = str(
                obj.get(
                    class_key,
                    obj.get("category", "unknown"),
                )
            )

            if obj.get("class_id") is not None:
                class_id = int(obj["class_id"])

            elif class_name in class_name_to_id:
                class_id = int(
                    class_name_to_id[class_name]
                )

            else:
                print(
                    f"[Warning] 对象 {object_index} 的类别 "
                    f"{class_name!r} 没有对应 class_id，跳过。"
                )
                skipped_count += 1
                continue

            # ----------------------------------------------
            # 2.2 获取二值 mask
            # ----------------------------------------------
            mask = None

            # 情况一：对象中直接保存了 numpy/tensor mask
            if obj.get("mask") is not None:
                mask_value = obj["mask"]

                if isinstance(mask_value, torch.Tensor):
                    mask = (
                        mask_value
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )

                elif isinstance(mask_value, np.ndarray):
                    mask = mask_value.copy()

                else:
                    mask = np.asarray(mask_value)

            # 情况二：从 mask 图片读取
            elif obj.get("mask_path"):
                mask_path = Path(
                    obj["mask_path"]
                ).expanduser()

                if not mask_path.is_absolute():
                    mask_path = (
                        json_base_dir / mask_path
                    ).resolve()

                if not mask_path.is_file():
                    print(
                        f"[Warning] mask 文件不存在：{mask_path}"
                    )
                    skipped_count += 1
                    continue

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

            # 情况三：已经保存了 polygon
            elif obj.get("polygon"):
                polygon = np.asarray(
                    obj["polygon"],
                    dtype=np.float32,
                )

                if (
                    polygon.ndim != 2
                    or polygon.shape[1] != 2
                    or len(polygon) < 3
                ):
                    skipped_count += 1
                    continue

                line = self._polygon_to_yolo_line(
                    polygon=polygon,
                    class_id=class_id,
                    image_width=image_width,
                    image_height=image_height,
                )

                if line is not None:
                    output_lines.append(line)
                else:
                    skipped_count += 1

                continue

            if mask is None:
                print(
                    f"[Warning] 对象 {object_index} 没有 mask。"
                )
                skipped_count += 1
                continue

            mask = np.squeeze(mask)

            if mask.ndim != 2:
                print(
                    f"[Warning] 对象 {object_index} mask shape "
                    f"无效：{mask.shape}"
                )
                skipped_count += 1
                continue

            # 恢复到原图分辨率
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 兼容 0/1、0/255、概率 mask
            if mask.max() > 1.0:
                binary_mask = (
                    mask > 127
                ).astype(np.uint8)
            else:
                binary_mask = (
                    mask > 0.5
                ).astype(np.uint8)

            # ----------------------------------------------
            # 2.3 mask 转 polygon
            # ----------------------------------------------
            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            valid_contours = [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= min_area
                and len(contour) >= 3
            ]

            if not valid_contours:
                print(
                    f"[Warning] 对象 {object_index} 未提取到有效轮廓。"
                )
                skipped_count += 1
                continue

            if use_largest_contour:
                valid_contours = [
                    max(
                        valid_contours,
                        key=cv2.contourArea,
                    )
                ]

            for contour in valid_contours:
                perimeter = cv2.arcLength(
                    contour,
                    closed=True,
                )

                epsilon = max(
                    0.5,
                    simplify_ratio * perimeter,
                )

                simplified = cv2.approxPolyDP(
                    contour,
                    epsilon,
                    closed=True,
                )

                polygon = (
                    simplified
                    .reshape(-1, 2)
                    .astype(np.float32)
                )

                if len(polygon) < 3:
                    continue

                line = self._polygon_to_yolo_line(
                    polygon=polygon,
                    class_id=class_id,
                    image_width=image_width,
                    image_height=image_height,
                )

                if line is not None:
                    output_lines.append(line)

        # --------------------------------------------------
        # 3. 写入 YOLO segmentation TXT
        # --------------------------------------------------
        with txt_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            if output_lines:
                file.write("\n".join(output_lines))
                file.write("\n")

        result = {
            "txt_path": str(txt_path),
            "num_instances": len(output_lines),
            "num_skipped": skipped_count,
        }

        print("=" * 60)
        print(f"YOLO segmentation TXT：{txt_path}")
        print(f"写入实例数：{len(output_lines)}")
        print(f"跳过实例数：{skipped_count}")
        print("=" * 60)

        return result


    @staticmethod
    def _polygon_to_yolo_line(
        polygon,
        class_id,
        image_width,
        image_height,
    ):
        """
        将像素 polygon 转换为一行 YOLO segmentation 标签。
        """
        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or len(polygon) < 3
        ):
            return None

        normalized = polygon.copy()

        normalized[:, 0] /= float(image_width)
        normalized[:, 1] /= float(image_height)

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        # 去除连续重复点
        keep_indices = [0]

        for index in range(1, len(normalized)):
            if not np.allclose(
                normalized[index],
                normalized[keep_indices[-1]],
            ):
                keep_indices.append(index)

        normalized = normalized[keep_indices]

        if len(normalized) < 3:
            return None

        coordinates = " ".join(
            f"{value:.6f}"
            for point in normalized
            for value in point
        )

        return f"{int(class_id)} {coordinates}"

    def close(self):
        """
        释放当前图像缓存和 CUDA 显存。
        """
        try:
            self.reset_image()
        except Exception:
            pass

        self.original_image = None
        self.image_height = None
        self.image_width = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL bbox + Ultralytics SAM3 mask"
    )

    parser.add_argument(
        "--image",
        default=r"\\158.132.186.40\isds\huilin\tp\20260617_113609.png",
        help="输入图像路径",
    )

    parser.add_argument(
        "--qwen-json",
        default=r"\\158.132.186.40\isds\huilin\tp\traffic_sign_result2.json",
        help="Qwen3-VL 输出 JSON 路径",
    )

    parser.add_argument(
        "--model",
        default=r"llm_tools/vllm/sam3.pt",
        help="sam3.pt 路径",
    )

    parser.add_argument(
        "--output-dir",
        default="./sam3_results",
        help="输出目录",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="例如 cuda:0、cuda:1 或 cpu",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.02,
        help="Qwen bbox 向外扩张比例",
    )

    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {
        "conf": args.conf,
        "task": "segment",
        "mode": "predict",
        "model": args.model,
        "device": args.device,
        "half": args.device != "cpu",
        "compile": False,
        "save": False,
        "show": False,
        "verbose": False,
    }

    predictor = TrafficSignSAM3Predictor(
        overrides=overrides
    )

    try:
        # 图像 encoder 只执行一次
        predictor.set_traffic_image(args.image)

        sam_result = predictor.segment_qwen_objects(
            qwen_result=args.qwen_json,
            output_dir=args.output_dir,
            expand_ratio=args.expand_ratio,
            mask_threshold=args.mask_threshold,
            overlay_alpha=0.45,
        )
        image_stem = Path(args.image).stem
        label_path = (
            Path(args.output_dir)
            / "labels"
            / f"{image_stem}.txt"
        )
        TRAFFIC_SIGN_CLASSES = TRAFFIC_SIGN_CLASSES = {
            "speed_limit_20": 0,
            "speed_limit_30": 1,
            "speed_limit_40": 2,
            "speed_limit_50": 3,
            "speed_limit_60": 4,
            "speed_limit_70": 5,
            "speed_limit_80": 6,
            "speed_limit_100": 7,
            "speed_limit_120": 8,
            "stop": 9,
            "no_entry": 10,
            "no_parking": 11,
            "no_stopping": 12,
            "pedestrian_crossing": 13,
            "turn_left": 14,
            "turn_right": 15,
            "straight_ahead": 16,
            "no_right_turn": 17,
            "unknown": 18,
        }
        export_result = predictor.export_yolo_seg_txt(
            sam_result=sam_result,
            txt_path=label_path,
            class_name_to_id=TRAFFIC_SIGN_CLASSES,
            class_key="specific_type",

            # 小于 20 像素的噪声区域不导出
            min_area=20.0,

            # 多边形简化程度
            simplify_ratio=0.001,

            # 一个交通标志只保留最大连通轮廓
            use_largest_contour=True,
        )

        print(
            json.dumps(
                sam_result,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        predictor.close()


if __name__ == "__main__":
    main()