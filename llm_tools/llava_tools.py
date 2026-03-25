import json
import os
import re
from pathlib import Path

import cv2
from tqdm import tqdm

id2cat_map = {
    0: "crack",
    1: "spalling",
    2: "moisture",
}


def yolo2llava_box(x_center, y_center, w, h, scale=1000, precision=3):
    """
    将 YOLO 的归一化中心坐标转换为 LLaVA 的离散化边界框 [ymin, xmin, ymax, xmax]
    """
    xmin = x_center - w / 2.0
    ymin = y_center - h / 2.0
    xmax = x_center + w / 2.0
    ymax = y_center + h / 2.0

    # 限制在 0-1 之间，防止越界
    xmin = max(0.0, min(1.0, xmin))
    ymin = max(0.0, min(1.0, ymin))
    xmax = max(0.0, min(1.0, xmax))
    ymax = max(0.0, min(1.0, ymax))

    # 映射到 [0, scale] 区间，LLaVA 常规使用 1000
    # b_xmin = int(xmin * scale)
    # b_ymin = int(ymin * scale)
    # b_xmax = int(xmax * scale)
    # b_ymax = int(ymax * scale)

    b_xmin = round(xmin, precision)
    b_ymin = round(ymin, precision)
    b_xmax = round(xmax, precision)
    b_ymax = round(ymax, precision)
    return f"[{b_xmin}, {b_ymin}, {b_xmax}, {b_ymax}]"


def level_judge(w, h):
    if w > 0.5 or h > 0.5:
        return "serious"
    elif w > 0.1 or h > 0.1:
        return "moderate"  # 拼写保持你的原样，或可改为 moderate
    else:
        return "minor"


def action_judge(cat_str, level_str):
    if level_str == "serious":
        return "repair"
    elif level_str == "medorate" or cat_str == "spalling":
        return "inspection"
    else:
        return "monitor"


def get_stem_list(txt_path):
    """
    读取 txt 文件，提取其中图片名称的 stem（去掉后缀和路径）
    """
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} does not exist.")
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        # 无论 txt 里写的是绝对路径还是仅文件名，Path(x).stem 都能提取出纯文件名
        return [Path(line.strip()).stem for line in f if line.strip()]


def convert_yolo_to_llava_table(
    img_dir, label_dir, train_txt, val_txt, output_train_json, output_val_json
):
    llava_train_dataset = []
    llava_val_dataset = []

    # 使用 set 可以加速后续的 in 判断
    train_stems = set(get_stem_list(train_txt))
    val_stems = set(get_stem_list(val_txt))

    img_list = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

    for img_name in tqdm(img_list, desc="Converting to LLaVA format"):
        img_stem = Path(img_name).stem
        label_path = os.path.join(label_dir, f"{img_stem}.txt")
        img_path = os.path.join(img_dir, img_name)

        # 定义问题：要求模型以表格形式输出检测结果和坐标
        human_prompt = (
            "<image>\n"
            "Please detect all defects in this image and output the results in a Markdown table format with columns."
        )

        # 读取 YOLO 标签并构建表格内容
        if not os.path.exists(label_path) or os.stat(label_path).st_size == 0:
            # 如果没有检测目标
            gpt_response = "There is no defect in this image."
        else:
            # 表格表头
            # 表格表头（水平格式）
            table_str = (
                "| Defect ID | Defect Type | Defect Level | Action | Bounding Box |\n"
                + "| --- | --- | --- | --- | --- |\n"
            )

            with open(label_path, "r") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])

                cat_str = id2cat_map.get(class_id, f"unknown_{class_id}")
                level_str = level_judge(w, h)
                action_str = action_judge(cat_str, level_str)
                bbox_str = yolo2llava_box(x_c, y_c, w, h)

                # 每一行记录一个缺陷
                table_str += (
                    f"| {idx} | {cat_str} | {level_str} | {action_str} | {bbox_str} |\n"
                )

            gpt_response = table_str.strip()

        # 组装 LLaVA 多轮对话格式
        llava_item = {
            "id": img_stem,
            "image": os.path.basename(img_path),
            "conversations": [
                {"from": "human", "value": human_prompt},
                {"from": "gpt", "value": gpt_response},
            ],
        }

        # 根据文件名将其划分到对应的集合中
        if img_stem in train_stems:
            llava_train_dataset.append(llava_item)
        elif img_stem in val_stems:
            llava_val_dataset.append(llava_item)
        else:
            # 如果既不在 train.txt 也不在 val.txt，可以根据需求选择 print 警告或直接跳过
            pass

    # 分别写入 JSON
    with open(output_train_json, "w", encoding="utf-8") as f:
        json.dump(llava_train_dataset, f, indent=4)
    print(
        f"\nTrain set conversion complete! Saved {len(llava_train_dataset)} items to {output_train_json}"
    )

    with open(output_val_json, "w", encoding="utf-8") as f:
        json.dump(llava_val_dataset, f, indent=4)
    print(
        f"Val set conversion complete! Saved {len(llava_val_dataset)} items to {output_val_json}"
    )


def parse_and_visualize_llava(
    image_path, llava_output_str, output_path="verify_result.jpg"
):
    """
    解析 LLaVA 输出的 Markdown 表格，提取类别和 [xmin, ymin, xmax, ymax] 坐标，并在图上绘制。
    """
    # 1. 读取图片获取真实宽高
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 错误: 无法读取图片 {image_path}")
        return

    h, w, _ = img.shape
    print(f"📸 图片读取成功，尺寸: 宽={w}, 高={h}")

    # 2. 逐行解析 Markdown 表格
    lines = llava_output_str.strip().split("\n")

    boxes = []

    # 正则表达式匹配 [0.xxx, 0.yyy, 0.zzz, 0.www]
    box_pattern = r"\[\s*(0\.\d+|0|1\.0+)\s*,\s*(0\.\d+|0|1\.0+)\s*,\s*(0\.\d+|0|1\.0+)\s*,\s*(0\.\d+|0|1\.0+)\s*\]"

    for line in lines:
        # 跳过表头和分割线
        if "Defect Type" in line or "---" in line:
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]

        # 确保这是一行有效的数据 (至少包含缺陷类型和坐标框)
        if len(parts) >= 4:
            defect_type = parts[0]
            box_str = parts[-1]  # 最后一列是 Bounding Box

            # 提取坐标
            match = re.search(box_pattern, box_str)
            if match:
                # 严格按照 xmin, ymin, xmax, ymax 解包
                xmin, ymin, xmax, ymax = map(float, match.groups())
                boxes.append((defect_type, xmin, ymin, xmax, ymax))

    if not boxes:
        print("⚠️ 未在文本中解析到有效的缺陷目标和坐标。")
        return

    print(f"🎯 共解析出 {len(boxes)} 个目标，开始绘制...")

    # 3. 绘制检测框和标签
    for idx, (defect_type, xmin, ymin, xmax, ymax) in enumerate(boxes):
        # 归一化坐标 -> 真实像素坐标
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)

        # 稍微做个越界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        print(f"  └ 发现 [{defect_type}]: 像素坐标(左上:{x1},{y1}, 右下:{x2},{y2})")

        # 画矩形框 (红色，线宽 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 画文字背景和文字，提高可读性
        label = f"{defect_type}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
        cv2.putText(
            img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # 4. 保存结果
    cv2.imwrite(output_path, img)
    print(f"\n✅ 绘制完成！可视化结果已保存至: {output_path}")


if __name__ == "__main__":
    # 基础路径配置
    base_dir = r"\\158.132.186.40\isds\huilin\bdd\open_source_data\cubit-det"

    img_directory = os.path.join(base_dir, "images")
    label_directory = os.path.join(base_dir, "labels")

    # 划分文件的路径
    train_txt_path = os.path.join(base_dir, "train.txt")
    val_txt_path = os.path.join(base_dir, "val.txt")

    # 输出的 JSON 路径
    output_train_file = os.path.join(base_dir, "llava_train.json")
    output_val_file = os.path.join(base_dir, "llava_val.json")

    convert_yolo_to_llava_table(
        img_dir=img_directory,
        label_dir=label_directory,
        train_txt=train_txt_path,
        val_txt=val_txt_path,
        output_train_json=output_train_file,
        output_val_json=output_val_file,
    )
