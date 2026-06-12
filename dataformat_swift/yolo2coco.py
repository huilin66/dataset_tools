import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm


def addCatItem(coco_data, category_dict, add_background=False):
    """
    添加类别信息到 COCO 数据中

    Args:
        coco_data: COCO 格式数据字典
        category_dict: 类别字典 {id: name}
        add_background: 是否添加背景类 (id=0)
    """
    if add_background:
        # 添加背景类，id=0
        background_item = {"supercategory": "none", "id": 0, "name": "background"}
        coco_data["categories"].append(background_item)
        # 其他类别 id 需要偏移 +1
        id_offset = 1
    else:
        id_offset = 0

    for k, v in category_dict.items():
        category_item = dict()
        category_item["supercategory"] = "none"
        category_item["id"] = int(k) + id_offset  # 根据是否添加背景调整 id
        category_item["name"] = v
        coco_data["categories"].append(category_item)


def addImgItem(coco_data, image_set, image_id, file_name, size):
    # image_id += 1
    image_item = dict()
    image_item["id"] = image_id
    image_item["file_name"] = file_name
    image_item["width"] = size[1]
    image_item["height"] = size[0]
    # image_item['license'] = None
    # image_item['flickr_url'] = None
    # image_item['coco_url'] = None
    # image_item['date_captured'] = str(datetime.today())
    coco_data["images"].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(
    coco_data, annotation_id, object_name, image_id, category_id, bbox, polygon=None
):
    annotation_item = dict()
    if polygon is None:
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        annotation_item["segmentation"] = seg
        annotation_item["area"] = bbox[2] * bbox[3]
    else:
        annotation_item["segmentation"] = [polygon.flatten().tolist()]
        area = poly2area(polygon)
        annotation_item["area"] = area
    annotation_item["iscrowd"] = 0
    annotation_item["ignore"] = 0
    annotation_item["image_id"] = image_id
    annotation_item["bbox"] = bbox
    annotation_item["category_id"] = category_id
    annotation_item["id"] = annotation_id
    coco_data["annotations"].append(annotation_item)


def xywhn2xywh(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.0) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.0) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    box = (xmin, ymin, w, h)
    return list(map(int, box))


def poly2area(polygon):
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly2xywh(polygon):
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]


def yolo2coco(
    image_path,
    anno_path,
    json_path,
    class_path,
    dst_img_dir=None,
    seg=False,
    skip_zero=False,
    add_background=False,  # 新增参数：是否添加背景类
):
    """
    将 YOLO 格式标注转换为 COCO 格式

    Args:
        image_path: 图像文件夹路径
        anno_path: 标注文件夹路径
        json_path: 输出 JSON 文件路径
        class_path: 类别文件路径 (每行一个类名)
        dst_img_dir: 目标图像文件夹路径 (可选)
        seg: 是否为分割标注
        skip_zero: 是否跳过类别 0 的标注
        add_background: 是否添加背景类 (id=0)，如果为 True，其他类别 id 从 1 开始
    """
    coco_data = dict()
    coco_data["images"] = []
    coco_data["annotations"] = []
    coco_data["categories"] = []

    # category_set = dict()
    image_set = set()

    image_id = 000000
    annotation_id = 0

    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)

    category_set = []
    with open(class_path, "r") as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_id = dict((k, v) for k, v in enumerate(category_set))

    # 修改：传递 add_background 参数
    addCatItem(coco_data, category_id, add_background)

    # 确定 category_id 的偏移量
    if add_background:
        id_offset = 1  # 背景类占用了 id=0，其他类别从 1 开始
    else:
        id_offset = 0  # 类别从 0 开始

    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path)]
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    for file in tqdm(files):
        if os.path.splitext(file)[-1] != ".txt" or "classes" in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img_path = images[index]
            if dst_img_dir is not None:
                dst_img_path = img_path.replace(image_dir, dst_img_dir)
                shutil.copy(img_path, dst_img_path)
            img = cv2.imread(img_path)
            shape = img.shape
            height, width = img.shape[:2]
            filename = images[index].split(os.sep)[-1]
            image_id += 1
            current_image_id = addImgItem(
                coco_data, image_set, image_id, filename, shape
            )
        else:
            continue
        with open(file, "r") as fid:
            for line in fid.readlines():
                line = [float(x) for x in line.strip().split()]
                category = int(line[0])

                # 如果需要跳过类别 0 (当 add_background=True 且原 YOLO 标注中 0 表示背景时)
                if skip_zero and category == 0:
                    continue

                category_name = category_id[category]

                # 调整 category_id，如果添加了背景类，需要 +1
                coco_category_id = category + id_offset

                if not seg:
                    bbox = xywhn2xywh((line[1], line[2], line[3], line[4]), shape)
                    addAnnoItem(
                        coco_data,
                        annotation_id,
                        category_name,
                        current_image_id,
                        coco_category_id,  # 使用调整后的 category_id
                        bbox,
                    )
                else:
                    polygon = np.array(line[1:]).reshape(-1, 2)
                    polygon[:, 0] = polygon[:, 0] * width
                    polygon[:, 1] = polygon[:, 1] * height
                    box = poly2xywh(polygon)
                    addAnnoItem(
                        coco_data,
                        annotation_id,
                        category_name,
                        current_image_id,
                        coco_category_id,  # 使用调整后的 category_id
                        box,
                        polygon,
                    )
                annotation_id += 1

    json.dump(coco_data, open(json_path, "w"))
    print("class nums:{}".format(len(coco_data["categories"])))
    print("image nums:{}".format(len(coco_data["images"])))
    print("bbox nums:{}".format(len(coco_data["annotations"])))
    print("categories: {}".format([c["name"] for c in coco_data["categories"]]))
    print("category ids: {}".format([c["id"] for c in coco_data["categories"]]))


if __name__ == "__main__":
    # ===== 使用示例 =====

    root_dir = r"E:\data\thesis\HTM\rgb_selected_3_p12_v41_el"
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels_update")
    class_path = os.path.join(root_dir, "class_update.txt")

    # # 情况1：不添加背景类，category_id 从 0 开始
    # json_path = os.path.join(root_dir, "instance_all_no_bg.json")
    # yolo2coco(
    #     image_dir,
    #     label_dir,
    #     json_path,
    #     class_path,
    #     add_background=False,  # 不添加背景
    # )

    # 情况2：添加背景类，category_id 从 1 开始 (0 为背景)
    json_path = os.path.join(root_dir, "instance_all_with_bg.json")
    yolo2coco(
        image_dir,
        label_dir,
        json_path,
        class_path,
        add_background=True,  # 添加背景
    )

    # 如果需要跳过 YOLO 标注中类别为 0 的框（当它们表示背景时）
    # json_path = os.path.join(root_dir, "instance_all_with_bg_skip.json")
    # yolo2coco(
    #     image_dir, label_dir, json_path, class_path,
    #     add_background=True,
    #     skip_zero=True  # 跳过原标注中类别为 0 的标注框
    # )
