import argparse
import os.path
import time
from typing import List, Union

import cv2
import json
import numpy as np
import onnxruntime as ort
import config as cfg
import logging
# from utils import auto_log_source, log_source
from pathlib import Path
import motpy as mot
import pandas as pd
from scipy.spatial.distance import cdist
from utils import create_empty_img

# region utils

def letterbox(img, new_shape=(640, 640), stride=32):
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        img (np.ndarray): Input image in BGR format.
        new_shape (Tuple[int, int]): Target shape as (height, width).

    Returns:
        (np.ndarray): Resized and padded image.
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left)

def crop_mask(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (np.ndarray): [n, h, w] array of masks.
        boxes (np.ndarray): [n, 4] array of bbox coordinates in relative point form.

    Returns:
        (np.ndarray): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescale bounding boxes from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (np.ndarray): The bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2).
        img0_shape (tuple): The shape of the target image, in the format of (height, width).
        ratio_pad (tuple): A tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not.

    Returns:
        (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2).
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes = boxes.copy()  # Create a copy to avoid modifying the original array
    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image shape (height, width).

    Args:
        boxes (np.ndarray): Bounding boxes to clip, in (x1, y1, x2, y2) format.
        shape (tuple): Image shape (height, width).

    Returns:
        (np.ndarray): Clipped bounding boxes.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def non_max_suppression_with_attributes(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        na=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
        end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes using NumPy.

    Args:
        prediction (np.ndarray): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
        classes (List[int]): Filter by class
        agnostic (bool): Class-agnostic NMS
        multi_label (bool): Allow multiple labels per box
        labels (List[List[Union[int, float, np.ndarray]]]): A priori labels
        max_det (int): Maximum number of detections
        nc (int): Number of classes
        max_time_img (float): Max time per image
        max_nms (int): Maximum boxes into NMS
        max_wh (int): Maximum box width and height
        in_place (bool): Modify prediction in place
        rotated (bool): Use rotated boxes
        end2end (bool): Model doesn't require NMS

    Returns:
        List[np.ndarray]: List of detections per image
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    if classes is not None:
        classes = np.array(classes)

    if prediction.shape[-1] == 6 + na or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[np.isin(pred[:, 5:6], classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4 - na  # number of masks
    ai = 4 + nc  # attribute start index
    mi = 4 + nc + na  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box

    prediction = np.transpose(prediction, (0, 2, 1))  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = np.concatenate((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)

    t = time.time()
    output = [np.zeros((0, 6 + nm + na))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]  # Slice first 4 columns
        cls = x[:, 4:4 + nc]  # Slice next nc columns
        attribute = x[:, 4 + nc:4 + nc + na]  # Slice next na columns
        mask = x[:, 4 + nc + na:]  # Slice remaining columns (nm)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), attribute, mask), axis=1)
            x = x[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[np.isin(x[:, 5:6], classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[np.argsort(-x[:, 4])[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores

        if rotated:
            pass
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = nms_np(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logging.warning(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def nms_np(boxes, scores, iou_threshold):
    """Pure NumPy NMS for axis-aligned boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance between two arrays of 2D points.

    Args:
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    Returns:
        (tuple): A tuple containing the indexes of the points with the shortest distance in arr1 and arr2 respectively.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (List[List]): Original segmentations in COCO's JSON file.
                               Each element is a list of coordinates, like [segmentation1, segmentation2,...].

    Returns:
        s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def masks2segments(masks, strategy="all"):
    """
    Convert masks to segments.

    Args:
        masks (torch.Tensor): The output of the model, which is a tensor of shape (batch_size, 160, 160).
        strategy (str): 'all' or 'largest'.

    Returns:
        (List): List of segment masks.
    """
    segments = []
    for x in masks.astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates.
    """

    # np.array (faster grouped)
    coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
    coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): The coords to be scaled of shape n,2.
        img0_shape (tuple): The shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): The ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1].
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (np.ndarray): A array of shape [mask_dim, mask_h, mask_w]
        masks_in (np.ndarray): A array of shape [n, mask_dim]
        bboxes (np.ndarray): A array of shape [n, 4]
        shape (tuple): Input image shape (h, w)
        upsample (bool): Whether to upsample the mask

    Returns:
        np.ndarray: Binary mask array of shape [n, h, w]
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape

    # Matrix multiplication equivalent
    masks = np.matmul(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)

    # Scale bboxes to mask dimensions
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, [0, 2]] *= width_ratio  # x1, x2
    downsampled_bboxes[:, [1, 3]] *= height_ratio  # y1, y2

    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        # Resize each mask individually
        resized_masks = np.zeros((masks.shape[0], ih, iw))
        for i, mask in enumerate(masks):
            resized_masks[i] = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
        masks = resized_masks

    return masks > 0.0  # Binary mask

# endregion


class Tracker:
    def __init__(self, iou_threshold=0.1, max_num_repeat=10):
        self.next_id = 0
        self.tracks = []  # 每个元素: {'id': int, 'bbox': [x1, y1, x2, y2], 'class': int}
        self.id_counts = {}
        self.iou_threshold = iou_threshold
        self.max_num_repeat = max_num_repeat

    def update(self, bboxes, class_ids, return_ids=True):
        updated_tracks = []
        current_frame_ids = []
        used_track_ids = set()

        for i in range(len(bboxes)):
            bbox = bboxes[i]
            class_id = class_ids[i]
            matched = False

            for track in self.tracks:
                if track['id'] in used_track_ids:
                    continue
                if track['class'] != class_id:
                    continue
                if self.id_counts.get(track['id'], 0) >= self.max_num_repeat:
                    continue

                iou = self.compute_iou(bbox, track['bbox'])
                if iou > self.iou_threshold:
                    updated_tracks.append({'id': track['id'], 'bbox': bbox, 'class': class_id})
                    current_frame_ids.append(track['id'])
                    used_track_ids.add(track['id'])  # ✅ 标记为已用
                    matched = True
                    break

            if not matched:
                updated_tracks.append({'id': self.next_id, 'bbox': bbox, 'class': class_id})
                current_frame_ids.append(self.next_id)
                self.next_id += 1

        self.tracks = updated_tracks
        if return_ids:
            return current_frame_ids
        return self.tracks


    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class GlobalIDManager:
    def __init__(self, max_frames=30, reid_threshold=0.5, max_num_repeat=10, same_camera_match=True):
        self.track_buffer = []
        self.max_frames = max_frames
        self.reid_threshold = reid_threshold
        self.frame_id_map = {}
        self.global_id_map = {}
        self.global_id_counts = {}
        self.max_num_repeat = max_num_repeat
        self.next_global_id = 800001
        self.same_camera_match = same_camera_match
        self.same_score_threshold = 0.95

    def _cosine_sim(self, a, b):
        return np.dot(a, b.T)

    def update(self, cam_id, timestamp, local_id, bbox, embedding):
        best_match = None
        best_score = -1
        frame_str = f'{timestamp}_{cam_id}'
        if frame_str in self.frame_id_map:
            self.frame_id = self.frame_id_map[frame_str]
        else:
            self.frame_id_map[frame_str] = len(self.frame_id_map)
            self.frame_id = self.frame_id_map[frame_str]

        current_image_key = f"{frame_str}_{local_id}"

        for track in self.track_buffer:
            # 是否匹配同一个摄像头
            if not self.same_camera_match and track['camera_id'] == cam_id:
                continue
            if abs(track['frame_id'] - self.frame_id) > self.max_frames:
                continue  # 超出缓存窗口
            if len(self.global_id_counts.get(track['global_id'], set())) >= self.max_num_repeat:
                continue  # 该全局ID已达到最大图像数量
            if embedding is not None and track['embedding'] is not None:
                sim = self._cosine_sim(embedding, track['embedding'])
                if sim > self.reid_threshold and sim > best_score:
                    best_match = track
                    best_score = sim
        if best_match:
            global_id = best_match['global_id']
            if best_score > self.same_score_threshold:
                best_match['frame_id'] = self.frame_id
                return global_id
            # 只有当这是一个新的图像时才增加计数
            if current_image_key not in self.global_id_counts.get(global_id, set()):
                self.global_id_counts[global_id] = self.global_id_counts.get(global_id, set())
                self.global_id_counts[global_id].add(current_image_key)
        else:
            global_id = self.next_global_id
            self.next_global_id += 1

            self.global_id_counts[global_id] = self.global_id_counts.get(global_id, set())
            self.global_id_counts[global_id].add(current_image_key)
            self.global_id_map[current_image_key] = global_id


        # 添加当前目标到缓存
        self.track_buffer.append({
            'frame_id': self.frame_id,
            'camera_id': cam_id,
            'local_id': local_id,
            'bbox': bbox,
            'embedding': embedding,
            'global_id': global_id
        })

        # 控制缓存大小（按帧）
        self._prune_buffer(self.frame_id)

        return global_id

    def _prune_buffer(self, current_frame_id):
        # 删除早于当前帧 max_frames 的条目
        self.track_buffer = [
            t for t in self.track_buffer
            if current_frame_id - t['frame_id'] <= self.max_frames
        ]


class FastReID_ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]
        return img

    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def extract(self, patch):
        input_tensor = self.preprocess(patch)
        feat = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0]
        feat = self.normalize(feat, axis=1)
        return feat


class InferenceService:
    _instance = None
    RISK_A_WEIGHT = cfg.RISK_A_WEIGHT
    RISK_B_WEIGHT = cfg.RISK_B_WEIGHT
    RISK_C_WEIGHT = cfg.RISK_C_WEIGHT
    RISK_D_WEIGHT = cfg.RISK_D_WEIGHT
    SMALL_SIZE_THRESHOLD = cfg.SMALL_SIZE_THRESHOLD
    OVERALL_RISK_THRESHOLD_MEDIUM = cfg.OVERALL_RISK_THRESHOLD_MEDIUM
    OVERALL_RISK_THRESHOLD_HIGH = cfg.OVERALL_RISK_THRESHOLD_HIGH

    @staticmethod
    def get_instance():
        if InferenceService._instance is None:
            InferenceService._instance = InferenceService(cfg.MODEL_ONNX_PATH, cfg.TEMP_RESULT_DIR, cfg.SHARE_DATA_DIR, cfg.REID_MODEL_ONNX_PATH, track=cfg.TRACKER)
        return InferenceService._instance

    def __init__(self, onnx_model, save_dir, share_data_dir, reid_model,
                 show=False, track='IoU', conf=cfg.CONF, iou=cfg.IOU, imgsz=cfg.IMGSZ,
                 classes=cfg.CLASSES, attributes=cfg.ATTRIBUTES, levels=cfg.LEVELS, color_palette=cfg.COLOR_PALETTE):
        self.session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider"], )
        self.save_dir = save_dir
        self.share_data_dir = share_data_dir
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = classes
        self.attributes = attributes
        self.levels = levels
        self.color_palette = color_palette
        self.conf = conf
        self.iou = iou
        self.nc = len(classes)
        self.na = len(attributes)
        self.nl = len(levels)
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_width, self.input_height = self.input_shape
        self.stride = 32
        self.show = show
        self.track = track
        self.current_name = None
        self.num_gap = 100000
        self.gnss_df = None
        self.min_dist_center = 100
        self.min_dist_box = 100
        self.id_map = [{} for _ in range(6)]
        self.risk_enlarge = 1.8

        if self.track == 'IoU':
            self.tracker = [Tracker() for _ in range(6)]
        elif self.track == 'MOT':
            self.tracker = [mot.MultiObjectTracker(
                dt=0.1,
                tracker_kwargs={
                    'max_staleness': 10,
                    # 'min_steps_alive': 3,
                    # 'similarity_threshold': 0.2
                }
            )
            for _ in range(6)
            ]
        else:
            self.tracker = None
        self._warmup()

        self.global_id_manager = GlobalIDManager()
        self.reid_model = FastReID_ONNX(reid_model)

    # @auto_log_source()
    def _warmup(self):
        dummy_input = np.zeros(self.input_shape, dtype=np.float32)
        logging.info('Warming up model...')
        self.session.run(None, {self.input_name: dummy_input})
        logging.info('Warming up model finished.')

        self.seg_dir = os.path.join(self.save_dir, 'seg')
        os.makedirs(self.seg_dir, exist_ok=True)

    # @auto_log_source()
    def _init_gnss(self, input_dir):
        logging.info('Initializing GNSS...')
        file_list = os.listdir(input_dir)
        for file_name in file_list:
            if file_name.lower().startswith('gnss'):
                file_path = os.path.join(input_dir, file_name)
                df = pd.read_csv(file_path, header=None, index_col=False, names=['timestamp', 'lat', 'lon', 'alt'],sep=' ')
                logging.info(f'Initializing GNSS success with {file_path}')
                break
            else:
                df = None
        if df is None:
            logging.warning('Initializing GNSS failed!')
        self.gnss_df = df
        self.gnss_df['timestamp'] = pd.to_numeric(self.gnss_df['timestamp'], errors='coerce')
        logging.info('GNSS INFO:')
        logging.info(f'GNSS START with:{self.gnss_df.head(5)}')
        logging.info(f'GNSS END with:{self.gnss_df.tail(5)}')

    # @auto_log_source()
    def gps_match(self):
        if self.gnss_df is None:
            self._init_gnss(self.share_data_dir)
        if self.gnss_df is not None:
            matched = self.gnss_df[self.gnss_df['timestamp'] == self.timestamp]  # 正确筛选条件
            if not matched.empty:
                row = matched.iloc[0]
                lat, lon = row['lat'], row['lon']
            else:
                logging.info(f'{self.timestamp} match failed. try {self.timestamp+1}')
                matched = self.gnss_df[self.gnss_df['timestamp'] == self.timestamp+1]  # 正确筛选条件
                if not matched.empty:
                    row = matched.iloc[0]
                    lat, lon = row['lat'], row['lon']
                else:
                    logging.info(f'{self.timestamp} match failed. try {self.timestamp - 1}')
                    matched = self.gnss_df[self.gnss_df['timestamp'] == self.timestamp - 1]  # 正确筛选条件
                    if not matched.empty:
                        row = matched.iloc[0]
                        lat, lon = row['lat'], row['lon']
                    else:
                        logging.info(f'{self.timestamp} match failed.')
                        matched = []
                        lat, lon = 0, 0
        else:
            matched = []
            lat, lon = 0, 0
        gnss = {'location': [lat, lon]}
        logging.info(f'match {self.timestamp} with {len(matched)} gnss data, get {gnss}')
        return gnss

    # @auto_log_source()
    def get_timestamp(self, current_name):
        logging.info('Analysis timestamp...')
        timestamp_str = Path(current_name).stem
        timestamp = int(timestamp_str)
        self.timestamp = timestamp
        logging.info(f'timestamp from {timestamp_str} to {timestamp}')

    # @log_source(file="onnx_depolyment.py", function="Inference ONNX")
    def infer_batch(self, img_path_list, save_name, save_dir=None, save_json=False):
        logging.info('image loading...')
        img_datas, img_infos = [], []
        for img_path in img_path_list:
            # if not os.path.exists(img_path):
            #     create_empty_img(cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1], img_path)
            img_data, img_info = self.preprocess(img_path, save_dir)
            img_datas.append(img_data)
            img_infos.append(img_info)

        self.get_timestamp(save_name)
        img_data_cat = np.concatenate(img_datas, axis=0)
        logging.info(f'image src with {img_infos[0]["src_data"].shape}...')
        logging.info(f'image infering with {img_data_cat.shape}...')
        outs = self.session.run(None, {self.session.get_inputs()[0].name: img_data_cat})
        logging.info('result processing...')
        results = []
        for i in range(len(img_datas)):
            result = self.postprocess_mask(img_datas[i], [out[i:i+1] for out in outs], img_infos[i], i)
            results.append(result)

        coord = self.gps_match()
        results.append(coord)

        result_file = self.save_seg_risk_batch6(results, save_name)
        if save_json:
            self.save_json_batch(results, img_infos)
        logging.info('finished!')
        return result_file

    # @auto_log_source()
    def preprocess(self, img_path, save_dir):
        """
        Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized.
        """

        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1]==1024:
            if img.shape[0]==615:
                pass
            else:
                logging.warning(f'resize img from {img.shape} to {(615, 1024)}...')
                img = cv2.resize(img, (1024, 615))
            self.min_dist_center = 100
            self.min_dist_box = 100
        elif img.shape[1]==4096:
            if img.shape[0]==2460:
                pass
            else:
                logging.warning(f'resize img from {img.shape} to {(2460, 4096)}...')
                img = cv2.resize(img, (4096, 2460))
            self.min_dist_center = 400
            self.min_dist_box = 400
        else:
            logging.warning(f'resize img from {img.shape} to {(608, 960)}...')
            img = cv2.resize(img, (960, 608))
            self.min_dist_center = 100
            self.min_dist_box = 100

        image_data, pad = letterbox(img, self.imgsz, self.stride)
        if not (image_data.shape[0]==608 and image_data.shape[1]==960):
            logging.warning(f'resize image_data from {image_data.shape} to {(608, 960)}...')
            image_data = cv2.resize(image_data, (960, 608))
        image_data = np.transpose(image_data, (2, 0, 1))[np.newaxis].astype(np.float32) / 255.0

        img_info = {}
        img_info['src_data'] = img
        img_info['current_name'] = os.path.basename(img_path).replace('cam_image_', '')
        img_info['img_height'], img_info['img_width'] = img.shape[:2]
        img_info['input_height'], img_info['input_width'] = self.input_height, self.input_width
        img_info['pad'] = pad
        img_info['save_dir'] = save_dir
        return image_data, img_info


    def get_global_id(self, i, object_result, img_info, boxs, cam_id, track_ids):
        x1, y1, x2, y2 = map(int, boxs[i])
        patch = img_info['src_data'][y1:y2, x1:x2]
        embedding = self.reid_model.extract(patch)
        global_id = self.global_id_manager.update(
            cam_id=cam_id,
            timestamp=self.timestamp,
            local_id=track_ids[i],
            bbox=boxs[i].tolist(),
            embedding=embedding
        )
        object_result['id'] = global_id

    def get_overall_risk(self, object_result):
        weights = [
            self.RISK_A_WEIGHT,
            self.RISK_B_WEIGHT,
            self.RISK_C_WEIGHT,
            self.RISK_D_WEIGHT]
        scores = [
            object_result['risk_a_score'],
            object_result['risk_b_score'],
            object_result['risk_c_score'],
            object_result['risk_d_score'],
        ]
        max_single_score = max(scores)
        weighted_risk_score = sum([
            s * w for s, w in zip(scores, weights)
        ])
        scores.append(weighted_risk_score)

        overall_risk_score = max(scores)

        if overall_risk_score <= self.OVERALL_RISK_THRESHOLD_MEDIUM or max_single_score<=self.OVERALL_RISK_THRESHOLD_MEDIUM:
            overall_risk_level = 0
        elif overall_risk_score <= self.OVERALL_RISK_THRESHOLD_HIGH:
            overall_risk_level = 1
        else:
            overall_risk_level = 2
        object_result['overall_risk_score'] = overall_risk_score
        object_result['overall_risk_level'] = overall_risk_level

    def postprocess_mask(self, prep_img, outs, img_info, cam_id):
        def get_int_id(id_map, track_id):
            if track_id not in id_map:
                id_map[track_id] = len(id_map)  # 新ID = 当前字典长度
            return id_map[track_id]
        cat_feature, preds, mask_coefficients, protos = outs[0], outs[1:4], outs[4], outs[5]
        preds = non_max_suppression_with_attributes(cat_feature, self.conf, self.iou, nc=self.nc, na=self.na)

        pred = preds[0]
        masks = process_mask(protos[0], pred[:, 6 + self.na:], pred[:, :4], prep_img.shape[2:], upsample=True)
        pred[:, :4] = scale_boxes(prep_img.shape[2:], pred[:, :4], [img_info['img_height'], img_info['img_width']])
        boxs = pred[:, :4]
        scores = pred[:, 4:5]
        class_ids = pred[:, 5:6]
        attributes_score = pred[:, 6:6 + self.na]
        attributes = np.floor(attributes_score * self.risk_enlarge * (self.nl)).astype(np.int64)
        attributes = np.clip(attributes, 0, self.nl - 1)  # Ensure no value exceeds 2

        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            boxs, scores, class_ids, attributes, attributes_score, masks = boxs[keep], scores[keep], class_ids[keep], attributes[keep], attributes_score[keep], masks[keep]

        boxs, scores, class_ids, attributes, attributes_score, masks = self.postprocess_result(boxs, scores, class_ids, attributes, attributes_score, masks, min_dist_center=self.min_dist_center, min_dist_box=self.min_dist_box)

        polygons_coords = masks2segments(masks)
        polygons_uvs = [scale_coords(masks.shape[1:], x, [img_info['img_height'], img_info['img_width']], normalize=True) for x in
                        polygons_coords]


        if self.track == 'IoU':
            track_ids_ = self.tracker[cam_id].update(boxs, class_ids)
            track_ids = [(cam_id + 1) * self.num_gap + get_int_id(self.id_map[cam_id], track_id) for track_id in track_ids_]
        elif self.track == 'MOT':
            detections = []
            for box, score, class_id in zip(boxs, scores, class_ids):
                det = mot.Detection(box=list(box), score=float(score), class_id=int(class_id))
                detections.append(det)
            self.tracker[cam_id].step(detections)
            active_tracks = self.tracker[cam_id].active_tracks()
            track_ids = [(cam_id+1)*self.num_gap + get_int_id(self.id_map[cam_id], track.id) for track in active_tracks]
        else:
            track_ids = [0] * len(boxs)
            
        if self.show:
            self.draw_result(img_info, boxs, scores, track_ids, class_ids, attributes, polygons_uvs)

        max_size = max(img_info['img_height'], img_info['img_width'])

        object_results = []
        for i in range(len(boxs)):
            box_width = int(boxs[i][2] - boxs[i][0])
            box_height = int(boxs[i][3] - boxs[i][1])
            box_area = box_width * box_height
            small_object = box_width < self.SMALL_SIZE_THRESHOLD*max_size and box_height < self.SMALL_SIZE_THRESHOLD*max_size
            if small_object:
                continue
            object_result = {
                'id': track_ids[i],
                'category': int(class_ids[i]),
                'score': float(scores[i]),
                'risk_a_score': float(attributes_score[i][2]),
                'risk_b_score': float(attributes_score[i][1]),
                'risk_c_score': float(attributes_score[i][3]),
                'risk_d_score': float(attributes_score[i][0]),
                'risk_a_value': int(attributes[i][2]),
                'risk_b_value': int(attributes[i][1]),
                'risk_c_value': int(attributes[i][3]),
                'risk_d_value': int(attributes[i][0]),
                'risk_a': int(attributes[i][2]>0),
                'risk_b': int(attributes[i][1]>0),
                'risk_c': int(attributes[i][3]>0),
                'risk_d': int(attributes[i][0]>0),
                'box': list(boxs[i]),
                'box_width': box_width,
                'box_height': box_height,
                'box_area': box_area,
                'small_object': int((boxs[i][2]-boxs[i][0])<self.SMALL_SIZE_THRESHOLD and (boxs[i][3]-boxs[i][1])<self.SMALL_SIZE_THRESHOLD)
            }
            if object_result['risk_a_value'] + object_result['risk_b_value'] + object_result['risk_c_value'] + object_result['risk_d_value'] > 0:
                self.get_global_id(i, object_result, img_info, boxs, cam_id, track_ids)
            # self.get_global_id(i, object_result, img_info, boxs, cam_id, track_ids)
            self.get_overall_risk(object_result)
            object_result['uvs'] = polygons_uvs[i].tolist()
            object_results.append(object_result)
        return object_results

    def postprocess_result(self, boxs, scores, class_ids, attributes, attributes_score, masks, min_dist_center=100, min_dist_box=100):

        def bbox_min_distance_matrix(boxes_a, boxes_b):
            """
            boxes_a: (N, 4)
            boxes_b: (M, 4)
            Returns: (N, M) distance matrix
            """

            # 拆分坐标
            xa1, ya1, xa2, ya2 = np.split(boxes_a, 4, axis=1)  # (N, 1)
            xb1, yb1, xb2, yb2 = np.split(boxes_b, 4, axis=1)  # (M, 1)

            # 广播计算 pairwise dx 和 dy
            dx = np.maximum(0, np.maximum(xb1.T, xa1) - np.minimum(xb2.T, xa2))  # (N, M)
            dy = np.maximum(0, np.maximum(yb1.T, ya1) - np.minimum(yb2.T, ya2))  # (N, M)

            # 欧几里得距离
            distances = np.sqrt(dx ** 2 + dy ** 2)

            return distances

        def merge_boxes(box1, box2):
            """合并两个边界框 [x1, y1, x2, y2]"""
            x1 = min(box1[0], box2[0])
            y1 = min(box1[1], box2[1])
            x2 = max(box1[2], box2[2])
            y2 = max(box1[3], box2[3])
            return [x1, y1, x2, y2]


        # take 5 class: projecting frame represent all frames

        wall_display_idx = self.classes.index('wall display')
        projecting_frame_idx = self.classes.index('projecting frame')
        projecting_display_idx = self.classes.index('projecting display')
        hanging_frame_idx = self.classes.index('hanging frame')
        hanging_display_idx = self.classes.index('hanging display')
        other_idx = self.classes.index('other')


        projecting_frame_ids = [i for i, class_id in enumerate(class_ids) if class_id == projecting_frame_idx]
        if len(projecting_frame_ids) == 0:
            return boxs, scores, class_ids, attributes, attributes_score, masks
        wall_display_ids = [i for i, class_id in enumerate(class_ids) if class_id == wall_display_idx]
        projecting_display_ids = [i for i, class_id in enumerate(class_ids) if class_id == projecting_display_idx]
        hanging_frame_ids = [i for i, class_id in enumerate(class_ids) if class_id == hanging_frame_idx]
        hanging_display_ids = [i for i, class_id in enumerate(class_ids) if class_id == hanging_display_idx]
        other_ids = [i for i, class_id in enumerate(class_ids) if class_id == other_idx]

        centers = np.stack([(boxs[:, 0] + boxs[:, 2]) / 2,
                            (boxs[:, 1] + boxs[:, 3]) / 2], axis=1)



        frame_centers = np.array([centers[i] for i in projecting_frame_ids]).reshape(-1, 2)
        frame_boxs = np.array([boxs[i] for i in projecting_frame_ids]).reshape(-1, 4)
        row_indices = np.arange(len(frame_centers))


        if len(projecting_display_ids) > 0:
            projecting_display_centers = np.array([centers[i] for i in projecting_display_ids]).reshape(-1, 2)
            projecting_display_boxs = np.array([boxs[i] for i in projecting_display_ids]).reshape(-1, 4)
            dist_projecting_center_matrix = cdist(frame_centers, projecting_display_centers)
            dist_projecting_box_matrix = bbox_min_distance_matrix(frame_boxs, projecting_display_boxs)
            dist_projecting = dist_projecting_center_matrix + dist_projecting_box_matrix
            projecting_match_indices = np.argmin(dist_projecting, axis=1)
            projecting_match_dist = np.min(dist_projecting, axis=1)
            projecting_match_center_dist = dist_projecting_center_matrix[row_indices, projecting_match_indices]
            projecting_match_box_dist = dist_projecting_box_matrix[row_indices, projecting_match_indices]
            projecting_match_center_dist_match = projecting_match_center_dist<min_dist_center
            projecting_match_box_dist_match = projecting_match_box_dist<min_dist_box
            projecting_match_dist_match = np.logical_and(projecting_match_center_dist_match,
                                                         projecting_match_box_dist_match)
        else:
            projecting_match_dist_match = np.array([False]*len(frame_centers))

        if len(hanging_display_ids) > 0:
            hanging_display_centers = np.array([centers[i] for i in hanging_display_ids]).reshape(-1, 2)
            hanging_display_boxs = np.array([boxs[i] for i in hanging_display_ids]).reshape(-1, 4)
            dist_hanging_center_matrix = cdist(frame_centers, hanging_display_centers)
            dist_hanging_box_matrix = bbox_min_distance_matrix(frame_boxs, hanging_display_boxs)
            dist_hanging = dist_hanging_center_matrix + dist_hanging_box_matrix
            hanging_match_indices = np.argmin(dist_hanging, axis=1)
            hanging_match_dist = np.min(dist_hanging, axis=1)
            hanging_match_center_dist = dist_hanging_center_matrix[row_indices, hanging_match_indices]
            hanging_match_box_dist = dist_hanging_box_matrix[row_indices, hanging_match_indices]
            hanging_match_center_dist_match = hanging_match_center_dist<min_dist_center
            hanging_match_box_dist_match = hanging_match_box_dist<min_dist_box
            hanging_match_dist_match = np.logical_and(hanging_match_center_dist_match,
                                                         hanging_match_box_dist_match)
        else:
            hanging_match_dist_match = np.array([False] * len(frame_centers))

        matched_result_ids = []

        new_boxs, new_scores, new_class_ids, new_attributes, new_attributes_score, new_masks = [], [], [], [], [], []
        for i, frame_idx in enumerate(projecting_frame_ids):
            frame_box = boxs[frame_idx]
            frame_score = scores[frame_idx]
            frame_attribute = attributes[frame_idx]
            frame_attribute_score = attributes_score[frame_idx]
            frame_mask = masks[frame_idx]

            if projecting_match_dist_match[i] and hanging_match_dist_match[i]:
                if projecting_match_dist[i] <= hanging_match_dist[i]:
                    match_id = projecting_display_ids[projecting_match_indices[i]]
                    match_class_id = projecting_display_idx
                else:
                    match_id = hanging_display_ids[hanging_match_indices[i]]
                    match_class_id = hanging_display_idx
            elif projecting_match_dist_match[i]:
                match_id = projecting_display_ids[projecting_match_indices[i]]
                match_class_id = projecting_display_idx
            elif hanging_match_dist_match[i]:
                match_id = hanging_display_ids[hanging_match_indices[i]]
                match_class_id = hanging_display_idx
            else:
                continue

            matched_result_ids.append(frame_idx)
            matched_result_ids.append(match_id)

            match_score = max(frame_score, scores[match_id])
            match_attribute = np.maximum(frame_attribute, attributes[match_id])
            match_attribute_score = np.maximum(frame_attribute_score, attributes_score[match_id])
            match_mask = np.logical_or(frame_mask, masks[match_id])
            match_box = merge_boxes(frame_box, boxs[match_id])

            new_boxs.append(match_box)
            new_scores.append(match_score)
            new_class_ids.append(match_class_id)
            new_attributes.append(match_attribute)
            new_attributes_score.append(match_attribute_score)
            new_masks.append(match_mask)

        for i in range(len(masks)):
            if i in matched_result_ids:
                continue
            else:
                new_boxs.append(boxs[i])
                new_scores.append(scores[i])
                new_class_ids.append(class_ids[i])
                new_attributes.append(attributes[i])
                new_attributes_score.append(attributes_score[i])
                new_masks.append(masks[i])
        new_boxs, new_scores, new_class_ids, new_attributes, new_attributes_score, new_masks = (
            np.array(new_boxs), np.array(new_scores), np.array(new_class_ids), np.array(new_attributes),
            np.array(new_attributes_score), np.array(new_masks))
        return new_boxs, new_scores, new_class_ids, new_attributes, new_attributes_score, new_masks

    def draw_result(self, img_info, boxs, scores, track_ids, class_ids, attributes, masks, alpha=0.5) -> None:
        img = img_info['src_data']
        current_name = img_info['current_name']
        save_dir = img_info['save_dir']

        colors = np.array([self.color_palette[int(i)] for i in class_ids]) / 255.0

        for box, score, class_id, track_id, attribute, color, mask in zip(boxs, scores, class_ids, track_ids, attributes, colors, masks):
            class_id = int(class_id)
            score = float(score)
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Create the label text with class name and score
            label = f"id:{track_id}; {self.classes[class_id]}: {score:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )
            polys = []
            for i in range(len(mask)):
                pos1 = float(mask[i][0]) * img.shape[1]
                pos2 = float(mask[i][1]) * img.shape[0]
                polys.append([pos1, pos2])
            polys = np.array(polys, np.int32)
            cv2.polylines(img, [polys], isClosed=True, color=color, thickness=2)
            mask = img.copy()
            cv2.fillPoly(mask, [polys], color=color)
            cv2.addWeighted(mask, alpha, img, 1 - alpha, 0, img)

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw attibutes:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[0])+15*len(attribute), int(box[1])+15*len(attribute))
            cv2.rectangle(img, p1, p2, (100, 100, 100))
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            for idx, att in enumerate(attribute):
                att_name = self.attributes[idx]
                att_level = self.levels[int(att)]
                att_label = f'{att_name}:{att_level}'
                pos = [x1, y1 + 15 * (idx + 1) - 10]
                cv2.putText(img, att_label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if save_dir is not None:
            save_path = os.path.join(save_dir, current_name)
            cv2.imwrite(save_path, img)
        return img

    # @auto_log_source()
    def save_json(self, outputs, img_info):
        save_dir = img_info['save_dir']
        current_name = img_info['current_name']
        if save_dir is not None:
            result_path = os.path.join(save_dir, Path(current_name).stem + ".json")
            with open(result_path, "w") as f:
                json.dump(outputs[1], f)
        else:
            logging.info(f'{save_dir} not found')

    # @auto_log_source()
    def save_json_batch(self, outputs, img_infos):
        save_dir = img_infos[0]['save_dir']
        current_name = img_infos[0]['current_name']
        if save_dir is not None:
            result_path = os.path.join(save_dir, Path(current_name).stem + ".json")
            results = [output[0] for output in outputs]
            with open(result_path, "w") as f:
                json.dump(results, f)
        else:
            logging.info(f'{save_dir} not found')

    # @auto_log_source()
    def save_seg_risk_batch6(self, results, save_name):
        logging.info(f'save {save_name}...')
        object_results_path = os.path.join(self.seg_dir, save_name)

        with open(object_results_path, "w") as f:
            json.dump(results, f)
        logging.info(f'finished!')
        return [object_results_path]


if __name__ == "__main__":
    # test_root = r'E:\data\202502_signboard\result\task0812\420'
    # input_list = [
    #     'input_1',
    #     'input_2',
    #     'input_3',
    #     'input_4',
    #     'input_5',
    #     'input_6',
    # ]

    test_root = r'E:\cp_dir\Val_set_test\Val_set'
    input_list = [
        'cam1',
        'cam2',
        'cam3',
        'cam4',
        'cam5',
        'cam6',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=r'best.onnx',
                        help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=os.path.join(test_root, input_list[0]),
                        help="Path to input image")
    parser.add_argument("--source1", type=str, default=os.path.join(test_root, input_list[0]),
                        help="Path to input image")
    parser.add_argument("--source2", type=str, default=os.path.join(test_root, input_list[1]),
                        help="Path to input image")
    parser.add_argument("--source3", type=str, default=os.path.join(test_root, input_list[2]),
                        help="Path to input image")
    parser.add_argument("--source4", type=str, default=os.path.join(test_root, input_list[3]),
                        help="Path to input image")
    parser.add_argument("--source5", type=str, default=os.path.join(test_root, input_list[4]),
                        help="Path to input image")
    parser.add_argument("--source6", type=str, default=os.path.join(test_root, input_list[5]),
                        help="Path to input image")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(test_root, "infer1003"),
                        help="Confidence threshold")
    parser.add_argument("--share_data_dir", type=str,
                        default=test_root,
                        help="Confidence threshold")
    parser.add_argument("--reid_model", type=str,
                        default=r'reid_model.onnx',
                        help="Path to reid_model ONNX model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    model = InferenceService(onnx_model=args.model, save_dir=args.save_dir, share_data_dir=args.share_data_dir, reid_model=args.reid_model, track='IoU',show=False)

    cam_list = [
        'DA5148683',
        'DA5324655',
        'DA4930148',
        'DA5324645',
        'DA5148680',
        'DA6102933',
    ]

    t1 = time.time()
    if os.path.isfile(args.source):
        results = model.infer(args.source)
        print(results)
    elif os.path.isdir(args.source):
        file_list = sorted(os.listdir(args.source))
        for i in range(len(file_list)):
            file_name = file_list[i]

            file_path_1 = os.path.join(args.source1, file_name)
            file_path_2 = os.path.join(args.source2, file_name.replace(cam_list[0], cam_list[1]))
            file_path_3 = os.path.join(args.source3, file_name.replace(cam_list[0], cam_list[2]))
            file_path_4 = os.path.join(args.source4, file_name.replace(cam_list[0], cam_list[3]))
            file_path_5 = os.path.join(args.source5, file_name.replace(cam_list[0], cam_list[4]))
            file_path_6 = os.path.join(args.source6, file_name.replace(cam_list[0], cam_list[5]))
            if os.path.exists(file_path_1) and os.path.exists(file_path_2) and os.path.exists(file_path_3) and os.path.exists(file_path_4) and os.path.exists(file_path_5) and os.path.exists(file_path_6):
                results = model.infer_batch([file_path_1, file_path_2, file_path_3, file_path_4, file_path_5, file_path_6],
                                            save_name=file_list[i].split("_")[-1].replace('.jpg', '.json'))

    t2 = time.time()
    print(f'inference time: {t2 - t1}')
