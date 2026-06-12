import glob
import json
import os
import shutil
from pathlib import Path
from re import L
from this import d

import cv2
import numpy as np
from tqdm import tqdm

COLOR_PALETTE = [
    (255, 42, 4),
    (79, 68, 255),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (0, 192, 38),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (104, 31, 17),
]


def time2name(input_dir):
    img_list = os.listdir(input_dir)
    time_list = [img_name.split(".")[0].split("_")[-1] for img_name in img_list]
    time2name = dict(zip(time_list, img_list))
    return time2name


def dir_vis(
    json_dir,
    input_dir1,
    input_dir2,
    input_dir3,
    input_dir4,
    input_dir5,
    input_dir6,
    vis_item="abcd",
    small_threshold=None,
):
    shutil.rmtree(input_dir1 + "_vis") if os.path.exists(input_dir1 + "_vis") else None
    shutil.rmtree(input_dir2 + "_vis") if os.path.exists(input_dir2 + "_vis") else None
    shutil.rmtree(input_dir3 + "_vis") if os.path.exists(input_dir3 + "_vis") else None
    shutil.rmtree(input_dir4 + "_vis") if os.path.exists(input_dir4 + "_vis") else None
    shutil.rmtree(input_dir5 + "_vis") if os.path.exists(input_dir5 + "_vis") else None
    shutil.rmtree(input_dir6 + "_vis") if os.path.exists(input_dir6 + "_vis") else None
    input1_time2name = time2name(input_dir1)
    input2_time2name = time2name(input_dir2)
    input3_time2name = time2name(input_dir3)
    input4_time2name = time2name(input_dir4)
    input5_time2name = time2name(input_dir5)
    input6_time2name = time2name(input_dir6)
    json_list = os.listdir(json_dir)
    for json_name in tqdm(json_list):
        json_path = os.path.join(json_dir, json_name)
        image_name = json_name.replace(".json", "")
        input1_path = os.path.join(
            input_dir1,
            input1_time2name[image_name] if image_name in input1_time2name else "None",
        )
        input2_path = os.path.join(
            input_dir2,
            input2_time2name[image_name] if image_name in input2_time2name else "None",
        )
        input3_path = os.path.join(
            input_dir3,
            input3_time2name[image_name] if image_name in input3_time2name else "None",
        )
        input4_path = os.path.join(
            input_dir4,
            input4_time2name[image_name] if image_name in input4_time2name else "None",
        )
        input5_path = os.path.join(
            input_dir5,
            input5_time2name[image_name] if image_name in input5_time2name else "None",
        )
        input6_path = os.path.join(
            input_dir6,
            input6_time2name[image_name] if image_name in input6_time2name else "None",
        )
        if (
            os.path.exists(input1_path)
            and os.path.exists(input2_path)
            and os.path.exists(input3_path)
            and os.path.exists(input4_path)
            and os.path.exists(input5_path)
            and os.path.exists(input6_path)
            and os.path.exists(json_path)
        ):
            files_vis(
                json_path,
                input1_path,
                input2_path,
                input3_path,
                input4_path,
                input5_path,
                input6_path,
                vis_item=vis_item,
                small_threshold=small_threshold,
            )
        else:
            print(f"{json_name} sample missing")


def files_vis(
    json_path,
    input1_path,
    input2_path,
    input3_path,
    input4_path,
    input5_path,
    input6_path,
    vis_item="abcd",
    small_threshold=None,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data1 = data[0]
    data2 = data[1]
    data3 = data[2]
    data4 = data[3]
    data5 = data[4]
    data6 = data[5]

    img1 = cv2.imread(input1_path)
    img1, with_defect = img_vis(
        img1, data1, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img1_vis_path = os.path.join(
            os.path.dirname(input1_path) + "_vis", os.path.basename(input1_path)
        )
        os.makedirs(os.path.dirname(img1_vis_path), exist_ok=True)
        cv2.imwrite(img1_vis_path, img1)

    img2 = cv2.imread(input2_path)
    img2, with_defect = img_vis(
        img2, data2, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img2_vis_path = os.path.join(
            os.path.dirname(input2_path) + "_vis", os.path.basename(input2_path)
        )
        os.makedirs(os.path.dirname(img2_vis_path), exist_ok=True)
        cv2.imwrite(img2_vis_path, img2)

    img3 = cv2.imread(input3_path)
    img3, with_defect = img_vis(
        img3, data3, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img3_vis_path = os.path.join(
            os.path.dirname(input3_path) + "_vis", os.path.basename(input3_path)
        )
        os.makedirs(os.path.dirname(img3_vis_path), exist_ok=True)
        cv2.imwrite(img3_vis_path, img3)

    img4 = cv2.imread(input4_path)
    img4, with_defect = img_vis(
        img4, data4, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img4_vis_path = os.path.join(
            os.path.dirname(input4_path) + "_vis", os.path.basename(input4_path)
        )
        os.makedirs(os.path.dirname(img4_vis_path), exist_ok=True)
        cv2.imwrite(img4_vis_path, img4)

    img5 = cv2.imread(input5_path)
    img5, with_defect = img_vis(
        img5, data5, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img5_vis_path = os.path.join(
            os.path.dirname(input5_path) + "_vis", os.path.basename(input5_path)
        )
        os.makedirs(os.path.dirname(img5_vis_path), exist_ok=True)
        cv2.imwrite(img5_vis_path, img5)

    img6 = cv2.imread(input6_path)
    img6, with_defect = img_vis(
        img6, data6, vis_item=vis_item, small_threshold=small_threshold
    )
    if with_defect:
        img6_vis_path = os.path.join(
            os.path.dirname(input6_path) + "_vis", os.path.basename(input6_path)
        )
        os.makedirs(os.path.dirname(img6_vis_path), exist_ok=True)
        cv2.imwrite(img6_vis_path, img6)


# def img_vis(img, results, vis_item="abcd", small_threshold=None):
#     height, width = img.shape[:2]
#     risk_sum = 0
#     for result in results:
#         id = result["id"]
#         score = float(result["score"])
#         cat = result["category"]
#         risk_a = result["risk_a"]
#         risk_b = result["risk_b"]
#         risk_c = result["risk_c"]
#         risk_d = result["risk_d"]
#         box_width = result["box_width"]
#         box_height = result["box_height"]
#         if small_threshold is not None and (
#             box_width < small_threshold and box_height < small_threshold
#         ):
#             continue
#         # small_object = result['small_object']
#         # if small_object:
#         #     continue
#         if vis_item == "all":
#             risk_sum = 1
#         else:
#             if "a" in vis_item:
#                 risk_sum += risk_a
#             if "b" in vis_item:
#                 risk_sum += risk_b
#             if "c" in vis_item:
#                 risk_sum += risk_c
#             if "d" in vis_item:
#                 risk_sum += risk_d
#         defect_str = f"{id}/{score:.4f}:"
#         if risk_a > 0:
#             defect_str += "riskA;"
#         if risk_b > 0:
#             defect_str += "riskB;"
#         if risk_c > 0:
#             defect_str += "riskC;"
#         if risk_d > 0:
#             defect_str += "riskD;"
#         uvs = result["uvs"]
#         color = COLOR_PALETTE[cat]
#         uvs = np.array(uvs, dtype=np.float32)
#         pixel_coords = (uvs * [width, height]).astype(np.int32)
#         cv2.polylines(
#             img,
#             [pixel_coords],
#             isClosed=True,
#             color=(color[0], color[1], color[2], 255),  # BGR+Alpha（红色实线）
#             thickness=2,
#             lineType=cv2.LINE_AA,
#         )
#         u_values = [uv[0] for uv in uvs]
#         v_values = [uv[1] for uv in uvs]
#         min_u = min(u_values)
#         min_v = min(v_values)
#         top_left_x = min_u * width
#         top_left_y = min_v * height
#         cv2.putText(
#             img,
#             defect_str,
#             (int(top_left_x), int(top_left_y) + 12),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             [0, 0, 0],
#             1,
#         )
#     if risk_sum == 0:
#         with_defect = False
#     else:
#         with_defect = True
#     # with_defect = True
#     return img, with_defect

import cv2
import numpy as np

cat_map = {
    0: "wall frame",
    1: "wall display",
    2: "projecting frame",
    3: "projecting display",
    4: "hanging frame",
    5: "hanging display",
    6: "other",
}


def img_vis(
    img,
    results,
    vis_item="abcd",
    small_threshold=None,
    alpha=0.3,
    font_scale=0.8,
    text_thickness=2,
):
    """
    img: 输入的图像 (numpy array)
    results: 缺陷/目标检测结果列表
    vis_item: 需要可视化的风险类别
    small_threshold: 小目标过滤阈值
    alpha: 多边形内部填充的透明度
    font_scale: 文字缩放大小
    text_thickness: 文字粗细
    """
    height, width = img.shape[:2]
    risk_sum = 0

    for result in results:
        id = result["id"]
        score = float(result["score"])
        cat = result["category"]
        cat_name = cat_map[int(cat)]
        risk_a = result["risk_a"]
        risk_b = result["risk_b"]
        risk_c = result["risk_c"]
        risk_d = result["risk_d"]
        box_width = result.get("box_width", 0)
        box_height = result.get("box_height", 0)

        if small_threshold is not None and (
            box_width < small_threshold and box_height < small_threshold
        ):
            continue

        if vis_item == "all":
            risk_sum = 1
        else:
            if "a" in vis_item:
                risk_sum += risk_a
            if "b" in vis_item:
                risk_sum += risk_b
            if "c" in vis_item:
                risk_sum += risk_c
            if "d" in vis_item:
                risk_sum += risk_d

        # ---------------- 关键修改点 ----------------
        # 把 cat (类别) 拼接到文本字符串的最前面
        defect_str = f"ID:{id} - {cat_name}({score:.2f}):"
        # --------------------------------------------

        if risk_a > 0:
            defect_str += "riskA;"
        if risk_b > 0:
            defect_str += "riskB;"
        if risk_c > 0:
            defect_str += "riskC;"
        if risk_d > 0:
            defect_str += "riskD;"

        uvs = result["uvs"]
        color = COLOR_PALETTE[cat]  # 如果color包含Alpha通道，请截取前3个元素 color[:3]

        uvs = np.array(uvs, dtype=np.float32)
        pixel_coords = (uvs * [width, height]).astype(np.int32)

        # 半透明多边形填充
        overlay = img.copy()
        cv2.fillPoly(
            overlay, [pixel_coords], color=color[:3] if len(color) > 3 else color
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 多边形边缘实线
        cv2.polylines(
            img,
            [pixel_coords],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        u_values = [uv[0] for uv in uvs]
        v_values = [uv[1] for uv in uvs]
        min_u = min(u_values)
        min_v = min(v_values)
        top_left_x = int(min_u * width)
        top_left_y = int(min_v * height)

        # 带背景色底板的文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            defect_str, font, font_scale, text_thickness
        )

        # 计算文本背景矩形的左上角和右下角坐标
        rect_p1 = (top_left_x, top_left_y - text_height - baseline - 6)
        rect_p2 = (top_left_x + text_width + 4, top_left_y)
        text_org = (top_left_x + 2, top_left_y - baseline - 3)

        # 边界保护
        if rect_p1[1] < 0:
            rect_p1 = (top_left_x, top_left_y)
            rect_p2 = (
                top_left_x + text_width + 4,
                top_left_y + text_height + baseline + 6,
            )
            text_org = (top_left_x + 2, top_left_y + text_height + 3)

        # 绘制背景框和文字
        cv2.rectangle(
            img,
            rect_p1,
            rect_p2,
            color=color[:3] if len(color) > 3 else color,
            thickness=-1,
        )
        cv2.putText(
            img,
            defect_str,
            text_org,
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    with_defect = risk_sum > 0
    return img, with_defect


def json2yolo(json_file):
    with open(json_file, "r") as f:
        datas = json.load(f)
    yolo_results = []
    for i in range(6):
        data = datas[i]
        lines = []
        for record in data:
            cat_id = record["category"]
            risk_a_value = record["risk_a_value"]
            risk_b_value = record["risk_b_value"]
            risk_c_value = record["risk_c_value"]
            risk_d_value = record["risk_d_value"]
            uvs = record["uvs"]
            uvs = [coord for pair in uvs for coord in pair]
            nums = [
                cat_id,
                4,
                risk_a_value,
                risk_b_value,
                risk_c_value,
                risk_d_value,
            ] + uvs
            nums_str = list(map(str, nums))
            line = " ".join(nums_str)
            lines.append(line + "\n")
        yolo_results.append(lines)
    return yolo_results


def json2yolo_track(
    json_file, img_size=[4096, 2456], mdet=False, seg=False, with_object_id=False
):
    with open(json_file, "r") as f:
        datas = json.load(f)
    yolo_results = []
    for i in range(6):
        data = datas[i]
        lines = []
        for record in data:
            cat_id = record["category"]
            nums = [cat_id]
            if mdet:
                risk_a_value = record["risk_a_value"]
                risk_b_value = record["risk_b_value"]
                risk_c_value = record["risk_c_value"]
                risk_d_value = record["risk_d_value"]
                nums += [4, risk_a_value, risk_b_value, risk_c_value, risk_d_value]
            if seg:
                polygon = record["uvs"]
                polygon = [item for sublist in polygon for item in sublist]
                nums += polygon
            else:
                box = record["box"]
                x1, y1, x2, y2 = (
                    box[0] / img_size[0],
                    box[1] / img_size[1],
                    box[2] / img_size[0],
                    box[3] / img_size[1],
                )
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                box = [x_center, y_center, width, height]
                nums += box
            if with_object_id:
                obj_id = record["id"]
                nums += [obj_id]

            nums_str = list(map(str, nums))
            line = " ".join(nums_str)
            lines.append(line + "\n")
        yolo_results.append(lines)
    return yolo_results


def write2txt(file_path, lines):
    with open(file_path, "w") as f:
        f.writelines(lines)


def get_stem2img(img_dir):
    img_list = os.listdir(img_dir)
    stem_list = [Path(img_name).stem.split("_")[-1] for img_name in img_list]
    stem2img_dict = dict(zip(stem_list, img_list))
    return stem2img_dict


def esimage_merge(input_dir):
    json_dir = os.path.join(input_dir, "output")
    yolo_dir = os.path.join(input_dir, "yolo_dataset")
    image_dir = os.path.join(yolo_dir, "images")
    print(f"reset {image_dir}...")
    shutil.rmtree(image_dir) if os.path.exists(image_dir) else None
    os.makedirs(image_dir, exist_ok=True)

    input1_stem2img = get_stem2img(os.path.join(input_dir, "input_1"))
    input2_stem2img = get_stem2img(os.path.join(input_dir, "input_2"))
    input3_stem2img = get_stem2img(os.path.join(input_dir, "input_3"))
    input4_stem2img = get_stem2img(os.path.join(input_dir, "input_4"))
    input5_stem2img = get_stem2img(os.path.join(input_dir, "input_5"))
    input6_stem2img = get_stem2img(os.path.join(input_dir, "input_6"))

    json_list = os.listdir(json_dir)
    for json_name in tqdm(json_list):
        timestamp_stem = Path(json_name).stem
        input1_name = (
            input1_stem2img[timestamp_stem]
            if timestamp_stem in input1_stem2img
            else "None"
        )
        input2_name = (
            input2_stem2img[timestamp_stem]
            if timestamp_stem in input2_stem2img
            else "None"
        )
        input3_name = (
            input3_stem2img[timestamp_stem]
            if timestamp_stem in input3_stem2img
            else "None"
        )
        input4_name = (
            input4_stem2img[timestamp_stem]
            if timestamp_stem in input4_stem2img
            else "None"
        )
        input5_name = (
            input5_stem2img[timestamp_stem]
            if timestamp_stem in input5_stem2img
            else "None"
        )
        input6_name = (
            input6_stem2img[timestamp_stem]
            if timestamp_stem in input6_stem2img
            else "None"
        )

        input1_path = os.path.join(input_dir, "input_1", input1_name)
        input2_path = os.path.join(input_dir, "input_2", input2_name)
        input3_path = os.path.join(input_dir, "input_3", input3_name)
        input4_path = os.path.join(input_dir, "input_4", input4_name)
        input5_path = os.path.join(input_dir, "input_5", input5_name)
        input6_path = os.path.join(input_dir, "input_6", input6_name)

        if os.path.exists(input1_path):
            shutil.copy(input1_path, os.path.join(image_dir, input1_name))
        if os.path.exists(input2_path):
            shutil.copy(input2_path, os.path.join(image_dir, input2_name))
        if os.path.exists(input3_path):
            shutil.copy(input3_path, os.path.join(image_dir, input3_name))
        if os.path.exists(input4_path):
            shutil.copy(input4_path, os.path.join(image_dir, input4_name))
        if os.path.exists(input5_path):
            shutil.copy(input5_path, os.path.join(image_dir, input5_name))
        if os.path.exists(input6_path):
            shutil.copy(input6_path, os.path.join(image_dir, input6_name))


# def esresult2yolo(input_dir):
#     json_dir = os.path.join(input_dir, 'output')
#     yolo_dir = os.path.join(input_dir, 'yolo_dataset')
#     image_dir = os.path.join(yolo_dir, 'images')
#     label_dir = os.path.join(yolo_dir, 'labels')
#     print(f'reset {label_dir}...')
#     shutil.rmtree(label_dir) if os.path.exists(label_dir) else None
#     os.makedirs(label_dir, exist_ok=True)

#     input1_stem2img = get_stem2img(os.path.join(input_dir, 'input_1'))
#     input2_stem2img = get_stem2img(os.path.join(input_dir, 'input_2'))
#     input3_stem2img = get_stem2img(os.path.join(input_dir, 'input_3'))
#     input4_stem2img = get_stem2img(os.path.join(input_dir, 'input_4'))
#     input5_stem2img = get_stem2img(os.path.join(input_dir, 'input_5'))
#     input6_stem2img = get_stem2img(os.path.join(input_dir, 'input_6'))

#     json_list = os.listdir(json_dir)
#     for json_name in tqdm(json_list):
#         yolo_results = json2yolo(os.path.join(json_dir, json_name))

#         timestamp_stem = Path(json_name).stem
#         input1_name = input1_stem2img[timestamp_stem] if timestamp_stem in input1_stem2img else 'None'
#         input2_name = input2_stem2img[timestamp_stem] if timestamp_stem in input2_stem2img else 'None'
#         input3_name = input3_stem2img[timestamp_stem] if timestamp_stem in input3_stem2img else 'None'
#         input4_name = input4_stem2img[timestamp_stem] if timestamp_stem in input4_stem2img else 'None'
#         input5_name = input5_stem2img[timestamp_stem] if timestamp_stem in input5_stem2img else 'None'
#         input6_name = input6_stem2img[timestamp_stem] if timestamp_stem in input6_stem2img else 'None'

#         input1_path = os.path.join(image_dir, input1_name)
#         input2_path = os.path.join(image_dir, input2_name)
#         input3_path = os.path.join(image_dir, input3_name)
#         input4_path = os.path.join(image_dir, input4_name)
#         input5_path = os.path.join(image_dir, input5_name)
#         input6_path = os.path.join(image_dir, input6_name)

#         if os.path.exists(input1_path):
#             write2txt(os.path.join(label_dir, Path(input1_name).stem+'.txt'), yolo_results[0])
#         if os.path.exists(input2_path):
#             write2txt(os.path.join(label_dir, Path(input2_name).stem+'.txt'), yolo_results[1])
#         if os.path.exists(input3_path):
#             write2txt(os.path.join(label_dir, Path(input3_name).stem+'.txt'), yolo_results[2])
#         if os.path.exists(input4_path):
#             write2txt(os.path.join(label_dir, Path(input4_name).stem+'.txt'), yolo_results[3])
#         if os.path.exists(input5_path):
#             write2txt(os.path.join(label_dir, Path(input5_name).stem+'.txt'), yolo_results[4])
#         if os.path.exists(input6_path):
#             write2txt(os.path.join(label_dir, Path(input6_name).stem+'.txt'), yolo_results[5])


def esresult2yolo(
    input_dir, img_size=[4096, 2456], mdet=False, seg=False, with_object_id=False
):
    json_dir = os.path.join(input_dir, "output")
    # json_dir = os.path.join(input_dir, 'infer', 'seg')
    yolo_dir = os.path.join(input_dir, "yolo_dataset")
    image_dir = os.path.join(yolo_dir, "images")
    label_dir = os.path.join(yolo_dir, "labels")
    print(f"reset {label_dir}...")
    shutil.rmtree(label_dir) if os.path.exists(label_dir) else None
    os.makedirs(label_dir, exist_ok=True)

    input1_stem2img = get_stem2img(os.path.join(input_dir, "input_1"))
    input2_stem2img = get_stem2img(os.path.join(input_dir, "input_2"))
    input3_stem2img = get_stem2img(os.path.join(input_dir, "input_3"))
    input4_stem2img = get_stem2img(os.path.join(input_dir, "input_4"))
    input5_stem2img = get_stem2img(os.path.join(input_dir, "input_5"))
    input6_stem2img = get_stem2img(os.path.join(input_dir, "input_6"))

    json_list = os.listdir(json_dir)
    for json_name in tqdm(json_list):
        yolo_results = json2yolo_track(
            os.path.join(json_dir, json_name), img_size, mdet, seg, with_object_id
        )

        timestamp_stem = Path(json_name).stem
        input1_name = (
            input1_stem2img[timestamp_stem]
            if timestamp_stem in input1_stem2img
            else "None"
        )
        input2_name = (
            input2_stem2img[timestamp_stem]
            if timestamp_stem in input2_stem2img
            else "None"
        )
        input3_name = (
            input3_stem2img[timestamp_stem]
            if timestamp_stem in input3_stem2img
            else "None"
        )
        input4_name = (
            input4_stem2img[timestamp_stem]
            if timestamp_stem in input4_stem2img
            else "None"
        )
        input5_name = (
            input5_stem2img[timestamp_stem]
            if timestamp_stem in input5_stem2img
            else "None"
        )
        input6_name = (
            input6_stem2img[timestamp_stem]
            if timestamp_stem in input6_stem2img
            else "None"
        )

        input1_path = os.path.join(image_dir, input1_name)
        input2_path = os.path.join(image_dir, input2_name)
        input3_path = os.path.join(image_dir, input3_name)
        input4_path = os.path.join(image_dir, input4_name)
        input5_path = os.path.join(image_dir, input5_name)
        input6_path = os.path.join(image_dir, input6_name)

        if os.path.exists(input1_path):
            write2txt(
                os.path.join(label_dir, Path(input1_name).stem + ".txt"),
                yolo_results[0],
            )
        if os.path.exists(input2_path):
            write2txt(
                os.path.join(label_dir, Path(input2_name).stem + ".txt"),
                yolo_results[1],
            )
        if os.path.exists(input3_path):
            write2txt(
                os.path.join(label_dir, Path(input3_name).stem + ".txt"),
                yolo_results[2],
            )
        if os.path.exists(input4_path):
            write2txt(
                os.path.join(label_dir, Path(input4_name).stem + ".txt"),
                yolo_results[3],
            )
        if os.path.exists(input5_path):
            write2txt(
                os.path.join(label_dir, Path(input5_name).stem + ".txt"),
                yolo_results[4],
            )
        if os.path.exists(input6_path):
            write2txt(
                os.path.join(label_dir, Path(input6_name).stem + ".txt"),
                yolo_results[5],
            )


def check_defect(input_label_path):
    with open(input_label_path) as f:
        data = f.readlines()
    defect_sum = 0
    for line in data:
        parts = line.strip().split(" ")
        risk_a = int(parts[2])
        risk_b = int(parts[3])
        risk_c = int(parts[4])
        risk_d = int(parts[5])
        defect_sum += risk_a + risk_b + risk_c + risk_d
        if defect_sum > 0:
            return True
    return False


def select_defect(input_dir, output_dir, ref_dir, src_image_dirname="images_select"):
    input_image_dir = os.path.join(input_dir, src_image_dirname)
    input_label_dir = os.path.join(input_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_image_dir)
    os.makedirs(output_label_dir)
    ref_label_dir = os.path.join(ref_dir, "labels")
    label_list = os.listdir(input_label_dir)
    for label_name in tqdm(label_list):
        ref_label_path = os.path.join(ref_label_dir, label_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        input_image_path = os.path.join(
            input_image_dir, label_name.replace(".txt", ".jpg")
        )
        if os.path.exists(ref_label_path) or not os.path.exists(input_image_path):
            continue

        output_label_path = os.path.join(output_label_dir, label_name)
        output_image_path = os.path.join(
            output_image_dir, label_name.replace(".txt", ".jpg")
        )
        result = check_defect(input_label_path)
        if result:
            shutil.copy(input_label_path, output_label_path)
            shutil.copy(input_image_path, output_image_path)
    print(
        f"select {len(os.listdir(output_image_dir))} from {len(os.listdir(input_image_dir))}"
    )


def yolo_to_mot(yolo_root, output_file, img_size=[4096, 2456]):
    """
    将YOLO格式检测结果转换为MOT格式的单个txt文件

    参数:
        yolo_root: 数据集根目录（包含images/和labels/文件夹）
        output_file: 输出的MOT格式文件路径（如mot_gt.txt）
    """
    img_dir = os.path.join(yolo_root, "images")
    label_dir = os.path.join(yolo_root, "labels")

    # 获取所有图像文件（确定帧数和顺序）
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    with open(output_file, "w") as f_out:
        for idx, img_path in enumerate(tqdm(img_files), 1):
            # 获取对应的标签文件路径
            img_name = Path(img_path).stem
            label_path = os.path.join(label_dir, f"{img_name}.txt")

            if not os.path.exists(label_path):
                continue

            # 读取YOLO格式标签
            with open(label_path, "r") as f_label:
                lines = f_label.readlines()

            # 解析每一行YOLO数据
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                class_id, x_center, y_center, width, height, group_id = map(
                    float, parts
                )

                # 转换为MOT格式的绝对坐标
                img_w, img_h = img_size  # 需替换为实际图像尺寸！
                x = (x_center - width / 2) * img_w  # 左上角x
                y = (y_center - height / 2) * img_h  # 左上角y
                w = width * img_w  # 宽度（像素）
                h = height * img_h  # 高度（像素）

                # MOT格式：<帧号>, <ID>, <x>, <y>, <w>, <h>, <置信度>, <类别>, <可见性>
                mot_line = f"{img_name},{int(group_id)},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,{int(class_id)},1\n"
                f_out.write(mot_line)


def rename_yolo_files(input_dir, output_dir, prefix_map=None):
    """
    将YOLO文件名重命名，原始名为DA4930148_20250812140329800，其中将_之前的前缀映射为数字
    并将images与labels中的所有文件复制到新文件夹

    参数:
        input_dir: 输入目录（包含images和labels子目录）
        output_dir: 输出目录
        prefix_map: 前缀映射字典，将原始前缀字符串映射为数字。如果为None，则自动分配数字
    """
    # 创建输出目录
    input_image_dir = os.path.join(input_dir, "images")
    input_label_dir = os.path.join(input_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 如果没有提供前缀映射，使用空字典
    if prefix_map is None:
        prefix_map = {}

    # 获取所有图像文件
    image_files = sorted(os.listdir(input_image_dir))

    # 用于存储自动生成的前缀映射
    auto_prefix_map = {}
    current_number = 1

    # 重命名并复制文件
    for image_file in tqdm(image_files):
        # 获取文件名和扩展名
        base_name, ext = os.path.splitext(image_file)

        # 提取前缀和时间戳部分（假设格式为DA4930148_20250812140329800）
        parts = base_name.split("_")
        if len(parts) >= 2:
            prefix = parts[0]
            timestamp = parts[1]
        else:
            prefix = base_name  # 如果没有下划线，使用原文件名作为前缀
            timestamp = ""

        # 获取前缀对应的数字
        if prefix in prefix_map:
            prefix_number = prefix_map[prefix]
        else:
            # 如果前缀不在映射中，自动分配一个数字
            if prefix not in auto_prefix_map:
                auto_prefix_map[prefix] = current_number
                current_number += 1
            prefix_number = auto_prefix_map[prefix]

        # 新文件名格式: 数字_timestamp.ext
        new_base_name = (
            f"{prefix_number:01d}_{timestamp}" if timestamp else f"{prefix_number:01d}"
        )
        new_image_name = new_base_name + ext
        new_label_name = new_base_name + ".txt"

        # 复制图像文件
        src_image_path = os.path.join(input_image_dir, image_file)
        dst_image_path = os.path.join(output_image_dir, new_image_name)
        shutil.copy2(src_image_path, dst_image_path)

        # 复制标签文件（如果存在）
        src_label_path = os.path.join(input_label_dir, base_name + ".txt")
        dst_label_path = os.path.join(output_label_dir, new_label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

    # 打印自动生成的前缀映射
    if auto_prefix_map:
        print(f"自动生成的前缀映射: {auto_prefix_map}")

    print(f"重命名完成: 共处理{len(image_files)}个文件，已保存到{output_dir}")


def select_topk_timestamp_files(input_dir, output_dir, k=5):
    """
    根据文件名中的时间戳选择前5个时间戳的所有数据，并复制到新的文件夹

    参数:
        input_dir: 输入目录（包含images和labels子目录）
        output_dir: 输出目录
    """
    # 创建输出目录
    input_image_dir = os.path.join(input_dir, "images")
    input_label_dir = os.path.join(input_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有图像文件并提取时间戳
    image_files = os.listdir(input_image_dir)
    timestamp_file_map = {}

    for image_file in image_files:
        base_name, ext = os.path.splitext(image_file)
        parts = base_name.split("_")
        if len(parts) >= 2:
            timestamp = parts[1]
            # 确保时间戳是数字
            if timestamp.isdigit():
                if timestamp not in timestamp_file_map:
                    timestamp_file_map[timestamp] = []
                timestamp_file_map[timestamp].append(image_file)

    # 按时间戳排序并选择前5个
    sorted_timestamps = sorted(timestamp_file_map.keys())
    topk_timestamps = sorted_timestamps[:k]

    # 复制选中的文件
    total_files = 0
    for timestamp in topk_timestamps:
        files = timestamp_file_map[timestamp]
        for image_file in tqdm(files, desc=f"处理时间戳 {timestamp}"):
            # 复制图像文件
            base_name, ext = os.path.splitext(image_file)
            src_image_path = os.path.join(input_image_dir, image_file)
            dst_image_path = os.path.join(output_image_dir, image_file)
            shutil.copy2(src_image_path, dst_image_path)

            # 复制标签文件（如果存在）
            src_label_path = os.path.join(input_label_dir, base_name + ".txt")
            dst_label_path = os.path.join(output_label_dir, base_name + ".txt")
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)

            total_files += 1

    print(
        f"选择完成: 共处理{total_files}个文件，来自{len(topk_timestamps)}个时间戳，已保存到{output_dir}"
    )
    print(f"选中的时间戳: {topk_timestamps}")


def rename_yolo_files(input_dir, output_dir, prefix_map=None):
    """
    将YOLO文件名重命名，原始名为DA4930148_20250812140329800，其中将_之前的前缀映射为数字
    并将images与labels中的所有文件复制到新文件夹

    参数:
        input_dir: 输入目录（包含images和labels子目录）
        output_dir: 输出目录
        prefix_map: 前缀映射字典，将原始前缀字符串映射为数字。如果为None，则自动分配数字
    """
    # 创建输出目录
    input_image_dir = os.path.join(input_dir, "images")
    input_label_dir = os.path.join(input_dir, "labels")
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels")
    shutil.rmtree(output_dir) if os.path.exists(output_dir) else None
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 如果没有提供前缀映射，使用空字典
    if prefix_map is None:
        prefix_map = {}

    # 获取所有图像文件
    image_files = sorted(os.listdir(input_image_dir))

    # 用于存储自动生成的前缀映射
    auto_prefix_map = {}
    current_number = 1

    # 重命名并复制文件
    for image_file in tqdm(image_files):
        # 获取文件名和扩展名
        base_name, ext = os.path.splitext(image_file)

        # 提取前缀和时间戳部分（假设格式为DA4930148_20250812140329800）
        parts = base_name.split("_")
        if len(parts) >= 2:
            prefix = parts[0]
            timestamp = parts[1]
        else:
            prefix = base_name  # 如果没有下划线，使用原文件名作为前缀
            timestamp = ""

        # 获取前缀对应的数字
        if prefix in prefix_map:
            prefix_number = prefix_map[prefix]
        else:
            # 如果前缀不在映射中，自动分配一个数字
            if prefix not in auto_prefix_map:
                auto_prefix_map[prefix] = current_number
                current_number += 1
            prefix_number = auto_prefix_map[prefix]

        # 新文件名格式: 数字_timestamp.ext
        new_base_name = (
            f"{prefix_number:01d}{timestamp}" if timestamp else f"{prefix_number:01d}"
        )
        new_image_name = new_base_name + ext
        new_label_name = new_base_name + ".txt"

        # 复制图像文件
        src_image_path = os.path.join(input_image_dir, image_file)
        dst_image_path = os.path.join(output_image_dir, new_image_name)
        shutil.copy2(src_image_path, dst_image_path)

        # 复制标签文件（如果存在）
        src_label_path = os.path.join(input_label_dir, base_name + ".txt")
        dst_label_path = os.path.join(output_label_dir, new_label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

    # 打印自动生成的前缀映射
    if auto_prefix_map:
        print(f"自动生成的前缀映射: {auto_prefix_map}")

    print(f"重命名完成: 共处理{len(image_files)}个文件，已保存到{output_dir}")


if __name__ == "__main__":
    pass
    data_dir = r"\\158.132.186.40\isds\huilin\isds\google_drive_data\task1008\01-04-56_PM\cameras"

    dir_vis(
        json_dir=os.path.join(data_dir, "infer1115"),
        input_dir1=os.path.join(data_dir, "camera1"),
        input_dir2=os.path.join(data_dir, "camera2"),
        input_dir3=os.path.join(data_dir, "camera3"),
        input_dir4=os.path.join(data_dir, "camera4"),
        input_dir5=os.path.join(data_dir, "camera5"),
        input_dir6=os.path.join(data_dir, "camera6"),
        vis_item="all",
    )
    # 示例用法
    # select_top5_timestamp_files(
    #     input_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3072_mseg_c6_0809',
    #     output_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\top5_timestamp'
    # )

    # rename_yolo_files(
    #     input_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3072_mseg_c6_0809',
    #     output_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\select'
    # )
    # rename_yolo_files(
    #     input_image_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3072_mseg_c6_0809\image',
    #     input_label_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3072_mseg_c6_0809\labels',
    #     output_dir=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\select'
    # )

    # dir_vis(
    #     json_dir=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_infer\seg',
    #     input_dir1=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_1',
    #     input_dir2=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_2',
    #     input_dir3=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_3',
    #     input_dir4=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_4',
    #     input_dir5=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_5',
    #     input_dir6=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_6',
    #     vis_item='all',
    #     small_threshold=32,
    # )

    # esresult2yolo(r'\\158.132.186.40\isds\huilin\isds\environ_sense_data\task0926\results\438_demo', mdet=True, seg=True, with_object_id=False)
    # select_defect(
    #     r'Y:\ZHL\isds\PS\task0725\results\290\yolo_dataset',
    #     r'Y:\ZHL\isds\PS\task0725\results\290\yolo_dataset_select',
    #     r'E:\data\202502_signboard\data_annotation\ps_data\task\merge_dir_0729',
    #     )
