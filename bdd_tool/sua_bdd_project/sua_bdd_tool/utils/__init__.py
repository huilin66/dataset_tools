import os
import json

def load_class_names(class_path):
    """加载 classes.txt，按行号对应类别ID"""
    if not class_path or not os.path.exists(class_path):
        print("⚠️ 未找到 classes.txt，将仅显示类别 ID")
        return None
    with open(class_path, "r", encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"📝 加载类别名称：")
    for i, name in enumerate(names):
        print(f"{i}: {name}")
    return names

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def yolo_write(det_results, save_path, result_format="string"):
    # 写入格式: class cx cy w h conf id
    with open(save_path, "w") as f:
        for det_result in det_results:
            if result_format == "string":
                f.write(det_result)
            elif result_format == "dict":
                cx, cy, w, h = det_result["cxcywh"]
                cls = det_result["cls"] if "cls" in det_result else 0
                conf = det_result["conf"] if "conf" in det_result else 1.0
                id = det_result["id"] if "id" in det_result else 0
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f} {id}\n")
            else:
                raise ValueError(f"Unsupported result_format: {result_format}")
