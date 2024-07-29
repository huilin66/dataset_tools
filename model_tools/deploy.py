import time
import yaml
import contextlib
import numpy as np
from preprocess import Compose
from onnxruntime import InferenceSession


# region tools

class PredictConfig(object):
    """set config of preprocess, postprocess and visualize
    Args:
        infer_config (str): path of infer_cfg.yml
    """

    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.preprocess_infos = yml_conf['Preprocess']
        self.infer_method = yml_conf.get("infer_method", 'Resize')
        self.slice_size = yml_conf.get("slice_size", [1280, 1280])
        self.overlap_ratio = yml_conf.get("overlap_ratio", [0.1, 0.1])
        self.label_list = yml_conf['label_list']
        self.color_list = yml_conf['color_list']
        self.num_class = len(self.label_list)
        self.score_threshold = yml_conf.get("score_threshold", 0.5)
        self.mask = yml_conf.get("mask", False)
        self.tracker = yml_conf.get("tracker", None)
        self.nms = yml_conf.get("NMS", None)
        self.fpn_stride = yml_conf.get("fpn_stride", None)
        self.param = yml_conf
        self.print_config()


    def print_config(self):
        print('-----------  Model Configuration -----------')
        # print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


class Profile(contextlib.ContextDecorator):
    """
    From Yolov8
    General-purpose Profile class for measuring elapsed time.
    Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        with Profile() as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 0.123456 s"
        ```
    """

    def __init__(self, t=0.0):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
        """
        self.t = t

    def __enter__(self):
        """Start timing."""
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        """Stop timing."""
        self.dt = time.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t:.6f} s"


def result2bbox(result):
    def xywh2xyxy(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    result = np.transpose(result, (0, 2, 1))
    bbox_coords = xywh2xyxy(result[..., :4])  # shape (batch, 8400, 4)
    class_scores = result[..., 4:]  # shape (batch, 8400, numclass)

    det_scores = np.max(class_scores, axis=-1, keepdims=True)  # shape (batch, 8400, 1)
    det_idx = np.argmax(class_scores, axis=-1)  # shape (batch, 8400)
    det_idx = np.expand_dims(det_idx, axis=-1)  # shape (batch, 8400, 1)

    final_output = np.concatenate([det_idx, det_scores, bbox_coords], axis=-1)
    return final_output


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets


def multiclass_nms(bboxs, num_classes, nms_match_threshold, nms_match_metric,
                   result_merge, result_merge_threshold):
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0: continue
        r = nms(bboxs[idxs, 1:], nms_match_threshold, nms_match_metric)
        filter_boxes = np.concatenate([np.full((r.shape[0], 1), c), r], 1)
        if result_merge:
            merged_boxes = merge_boxes(filter_boxes, overlap_threshold=result_merge_threshold)
        else:
            merged_boxes = filter_boxes
        final_boxes.append(merged_boxes)
    if len(final_boxes) > 0:
        final_boxes = np.concatenate(final_boxes, axis=0)
    else:
        final_boxes = np.zeros((0, 6))
    return final_boxes

def merge_boxes(bboxes, overlap_threshold):
    '''
    始终以第0个bbox为基准，与之后的bbox进行比较
    如果第0个bbox没有与之后的bbox有重叠，pop第0个bbox，并添加到最终结果中
    如果第0个bbox与之后的bbox有重叠，合并第0个bbox与匹配bbox，得到新bbox，删除第0个bbox与匹配bbox，将新bbox添加到list中，重新开始对比
    '''

    # 根据置信度对框进行排序
    sorted_indices = np.argsort(np.squeeze(bboxes[:, 1:2]))[::-1]
    sorted_bboxes = bboxes[sorted_indices]


    merged_bboxs = []

    # 遍历已排序的框
    while sorted_bboxes.shape[0] > 1:
        # 取出置信度最高的框
        bbox = sorted_bboxes[0]
        base_box = bbox[2:]

        matched_flag = False
        # 遍历剩余的框
        for idx, next_bbox in enumerate(sorted_bboxes[1:, :]):
            next_box = next_bbox[2:]
            # 计算两个框的重叠区域
            x1 = max(base_box[0], next_box[0])
            y1 = max(base_box[1], next_box[1])
            x2 = min(base_box[2], next_box[2])
            y2 = min(base_box[3], next_box[3])

            # 计算重叠区域的宽度和高度
            overlap_width = max(0, x2 - x1 + 1)
            overlap_height = max(0, y2 - y1 + 1)

            # 计算交并比
            overlap_area = overlap_width * overlap_height
            box_area = (next_box[2] - next_box[0] + 1) * (next_box[3] - next_box[1] + 1)
            overlap_ratio = overlap_area / float(box_area)

            # 如果重叠比例大于阈值，则合并框
            if overlap_ratio > overlap_threshold:
                matched_flag = True
                merged_box = np.array([bbox[0], bbox[1], x1, y1, x2, y2]).reshape(1, 6)
                break

        if matched_flag:
            bbox1 = sorted_bboxes[1:idx+1, :]
            bbox2 = sorted_bboxes[idx+2:, :]
            sorted_bboxes = np.concatenate([merged_box, bbox1, bbox2], axis=0)

        else:
            merged_bboxs.append(bbox)
            sorted_bboxes = sorted_bboxes[1:, :]

    if sorted_bboxes.shape[0] == 1:
        merged_bboxs.append(sorted_bboxes[0])
    elif sorted_bboxes.shape[0] > 1:
        print('shape error')
    else:
        pass
    if len(merged_bboxs) == 0:
        return np.zeros((0, 6))
    else:
        return np.array(merged_bboxs)
# endregion


def predict_image(predictor, img_path, infer_config, output_format):
    transforms = Compose(infer_config.preprocess_infos)

    inputs = transforms(img_path)
    inputs_name = [var.name for var in predictor.get_inputs()]
    if 'images' in inputs_name:
        inputs_dict = {'images': inputs['image'][None,]}
    else:
        inputs_dict = {k: inputs[k][None,] for k in inputs_name}

    outputs = predictor.run(output_names=None, input_feed=inputs_dict)

    result = np.array(outputs[1])
    bboxes = result2bbox(result)[0]
    bboxes = bboxes[bboxes[:, 1] > infer_config.score_threshold]

    # if output_format == 0:
    #     bboxes = np.array(outputs[0])
    #     bboxes = bboxes[bboxes[:, 1]>infer_config.score_threshold]
    # elif output_format == 1:
    #     result = np.array(outputs[1])
    #     bboxes = result2bbox(result)[0]
    #     bboxes = bboxes[bboxes[:, 1] > infer_config.score_threshold]
    # elif output_format == 2:
    #     result = np.array(outputs[0])
    #     bboxes = result2bbox(result)[0]
    #     bboxes = bboxes[bboxes[:, 1] > infer_config.score_threshold]
    # else:
    #     print('[%g] output format not supported' % output_format)

    bboxes[:, 2] /= inputs['scale_factor'][1]
    bboxes[:, 4] /= inputs['scale_factor'][1]
    bboxes[:, 3] /= inputs['scale_factor'][0]
    bboxes[:, 5] /= inputs['scale_factor'][0]

    final_bboxes = multiclass_nms(
        bboxes,
        infer_config.num_class,
        nms_match_threshold=infer_config.param['nms_match_threshold'],
        nms_match_metric=infer_config.param['nms_match_metric'],
        result_merge=infer_config.param['result_merge'],
        result_merge_threshold=infer_config.param['result_merge_threshold'],
    )
    return final_bboxes

def predict_images(weight_path, img_dir, config_path, output_format):
    predictor = InferenceSession(weight_path)
    infer_config = PredictConfig(config_path)