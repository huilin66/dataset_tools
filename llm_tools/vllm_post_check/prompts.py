"""Generic heavy YOLO prompts for Qwen/VLLM visual detection and refinement."""

from __future__ import annotations

import json


def class_list_text(classes: list[str]) -> str:
    return "\n".join(f"{idx}: {name}" for idx, name in enumerate(classes))


def task_context_text(task_type: str | None = None) -> str:
    task_type = str(task_type or "").strip()
    if not task_type:
        return "任务背景：通用视觉目标检测。"
    return f"任务背景：{task_type}。"


def _common_rules(classes: list[str], task_type: str | None = None) -> str:
    return f"""
{task_context_text(task_type)}

允许类别列表如下。你只能输出这些类别名称，不得自创类别：
{class_list_text(classes)}

你必须遵守以下原则：

1. 只依据图像中明确可见的视觉证据作出判断，不得猜测。
2. 只检测属于允许类别列表的目标；背景物体、相似但不属于任务目标的物体不要输出。
3. 如果目标模糊、遮挡严重、类别无法可靠判断，但仍明显属于允许类别中的某一类，可输出最可能类别并降低 confidence。
4. 如果无法确认目标属于允许类别，应忽略该目标。
5. 如果同一目标被分成多个可见部分，但明显属于同一个实例，应输出一个完整目标框，不要重复输出。
6. 如果图像中存在多个独立目标，必须逐个输出，不能合并成一个框。
7. 边界框应尽量完整覆盖目标主体，不要包含过多背景。
8. 不要把阴影、反光、运动模糊、压缩噪声、背景文字、背景图案或拍摄视角造成的形变当作目标。

坐标定义：

- 所有 bbox_norm_1000 均使用相对于当前输入图像宽高的 0 至 1000 归一化坐标。
- bbox_norm_1000 格式为 [x1, y1, x2, y2]。
- x1, y1 是左上角坐标；x2, y2 是右下角坐标。
- 坐标必须是整数。
- 必须满足 0 <= x1 < x2 <= 1000。
- 必须满足 0 <= y1 < y2 <= 1000。

confidence 定义：

- confidence 必须是 0 到 1 之间的连续小数，不是 0/1 二值标签。
- confidence 表示你基于视觉证据对“目标存在 + 类别正确 + 框位置合理”的综合确定性。
- 目标清晰、类别明确、框位置准确时可较高。
- 图像模糊、遮挡、类别相似、目标很小或框位置不确定时必须降低。
- 该数值是视觉判断置信度，不是经过校准的检测器概率。

输出要求：

- 必须严格输出合法 JSON 对象。
- 不要输出 Markdown。
- 不要输出代码块标记。
- 不要在 JSON 前后添加解释文字。
- 不要输出注释。
- 不要使用 NaN、Infinity、None 或其他非 JSON 值。
""".strip()


def full_image_prompt(classes: list[str], task_type: str | None = None) -> str:
    return f"""
你是一个严谨的视觉语言模型，需要执行通用 YOLO 风格目标检测任务。

你必须完成以下任务：

1. 检测图像中所有属于允许类别列表的可见目标；
2. 为每个目标输出准确边界框；
3. 判断每个目标的类别；
4. 为每个目标给出 0 到 1 的连续 confidence；
5. 简要说明判断依据；
6. 只依据图像中明确可见的视觉证据作出判断，不得猜测。

{_common_rules(classes, task_type)}

请严格按以下 JSON 格式输出：

{{
  "has_target": true,
  "detections": [
    {{
      "id": 1,
      "class_name": "allowed_class_name",
      "bbox_norm_1000": [120, 180, 310, 460],
      "confidence": 0.86,
      "keep": true,
      "reason": "简要说明直接可见的视觉依据"
    }}
  ]
}}

字段要求：

1. has_target
   - 如果存在至少一个有效目标，则为 true；
   - 如果没有有效目标，则为 false。

2. detections
   - 每个有效目标一个对象；
   - id 从 1 开始连续编号；
   - class_name 必须完全使用允许类别列表中的类别名称；
   - bbox_norm_1000 必须覆盖目标主体；
   - confidence 必须为 0 到 1 的连续小数；
   - keep 对有效检测必须为 true；
   - reason 只描述图像中能直接观察到的证据。

一致性规则：

1. has_target=false 时，detections 必须为 []。
2. detections 非空时，has_target 必须为 true。
3. 每个 detection 的 class_name 必须来自允许类别列表。
4. 每个 bbox_norm_1000 必须合法且位于 0 到 1000 范围内。
5. 不确定时降低 confidence；无法确认时不要输出。

如果图像中没有属于允许类别列表的目标，只返回：

{{
  "has_target": false,
  "detections": []
}}
""".strip()


def crop_refine_prompt(
    classes: list[str], candidate_class: str, task_type: str | None = None
) -> str:
    return f"""
你是一个严谨的视觉语言模型，需要对一个 YOLO 候选框裁剪图进行通用后处理复核。

原始 YOLO 候选类别为：{candidate_class}

你必须完成以下任务：

1. 判断当前裁剪图中是否确实存在一个属于允许类别列表的目标；
2. 判断原始 YOLO 候选类别是否正确；
3. 如果类别错误但目标仍属于允许类别列表，应修正为正确类别；
4. 如果目标不属于允许类别列表，或视觉证据不足，应拒绝该候选；
5. 给出 0 到 1 的连续 confidence；
6. 在需要重新定位时，输出目标在当前裁剪图中的 bbox_norm_1000。

{_common_rules(classes, task_type)}

请严格按以下 JSON 格式输出：

候选应保留时：

{{
  "has_target": true,
  "detections": [
    {{
      "id": 1,
      "class_name": "allowed_class_name",
      "bbox_norm_1000": [120, 180, 850, 900],
      "confidence": 0.82,
      "keep": true,
      "reason": "简要说明保留或修正类别的视觉依据"
    }}
  ]
}}

候选应拒绝时：

{{
  "has_target": false,
  "detections": [
    {{
      "id": 1,
      "class_name": "{candidate_class}",
      "bbox_norm_1000": [],
      "confidence": 0.76,
      "keep": false,
      "reason": "简要说明拒绝该候选的视觉依据"
    }}
  ]
}}

字段要求：

1. class_name
   - 保留候选时，必须使用允许类别列表中的类别名称；
   - 如果原始类别正确，保持原类别；
   - 如果原始类别错误但可判断正确类别，输出修正后的类别。

2. keep
   - true 表示候选应保留；
   - false 表示候选应删除；
   - 只有存在明确视觉证据时才设为 true。

3. bbox_norm_1000
   - 在 detect refinement 中必须给出目标在当前裁剪图中的 bbox_norm_1000；
   - 在 classification-only refinement 中可以给出 bbox，也可以输出 []；
   - 如果 keep=false，必须输出 []。

4. confidence
   - 必须是 0 到 1 的连续小数；
   - keep=true 时，表示保留和类别判断的确定性；
   - keep=false 时，表示拒绝该候选的确定性；
   - 不确定时应降低 confidence，不要用 0/1 代替连续判断。

一致性规则：

1. keep=false 时，has_target 必须为 false，bbox_norm_1000 必须为 []。
2. keep=true 时，has_target 必须为 true，class_name 必须来自允许类别列表。
3. 如果目标只是背景、阴影、反光、模糊纹理或与任务无关的物体，keep 必须为 false。
4. 如果裁剪图中有多个目标，优先复核位于裁剪图中心、最可能对应原始 YOLO 候选框的目标。
5. 不要因为原始 YOLO 给出了候选框就默认保留，必须重新根据图像证据判断。
""".strip()

PROMPT_VERSIONS = ("p1", "p2")


def p2_full_image_prompt(classes: list[str], task_type: str | None = None) -> str:
    """Exact prompt emitted by convert_full_detection.py."""
    del task_type
    return (
        "Detect every target belonging to the following classes: "
        f"{json.dumps(classes, ensure_ascii=False)}. "
        "Return only a JSON array. Each item must contain exactly "
        '"bbox_2d": [x1, y1, x2, y2] and "label". '
        "Return [] when no target exists."
    )


def p2_crop_refine_prompt(
    classes: list[str], candidate_class: str, task_type: str | None = None
) -> str:
    """Exact prompt emitted by convert_crop_classification.py."""
    del candidate_class, task_type
    return (
        "Classify the main target in this cropped image. "
        f"Choose exactly one label from {json.dumps(classes, ensure_ascii=False)}. "
        "Return only the label, without explanation or JSON."
    )

