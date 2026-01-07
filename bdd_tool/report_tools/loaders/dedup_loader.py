import os
import numpy as np
from loaders.yolo_loader import YoloLoader

class DedupLoader(YoloLoader):
    """
    专门加载 yolo_dedup.py 生成的 labels_dedup_fuse 数据
    格式: class cx cy w h conf id
    """
    def _yolo_norm_to_pixel(self, yolo_line, img_w, img_h):
        parts = yolo_line.strip().split()
        
        # 解析逻辑
        if len(parts) >= 7:
            # 标准 7 列格式
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            track_id = int(parts[6])
        elif len(parts) == 6:
            # 缺失 conf 的 6 列格式
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = 1.0 # 默认置信度
            track_id = int(parts[5])
        else:
            # 异常回退
            return super()._yolo_norm_to_pixel(yolo_line, img_w, img_h) + [-1]

        # 坐标转换 (Norm -> Pixel)
        x_center = xc * img_w
        y_center = yc * img_h
        width = w * img_w
        height = h * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 统一输出标准 7 维格式: [cls, conf, x1, y1, x2, y2, id]
        return [cls_id, conf, x1, y1, x2, y2, track_id]

    def load(self):
        # 复用父类逻辑，但因为 _yolo_norm_to_pixel 变了，
        # 返回的 detections numpy array 会多一列
        return super().load()