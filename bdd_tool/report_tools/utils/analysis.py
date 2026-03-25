# utils/analysis.py
import os
import numpy as np
from PIL import Image

# utils/analysis.py
import os
import numpy as np
from PIL import Image
from config import LEVELS_THRESHOLD # <--- 引入配置

def level_judge(box):
    """判断缺陷等级"""
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    # 使用配置中的阈值
    if w > LEVELS_THRESHOLD[1] or h > LEVELS_THRESHOLD[1]:
        return 'Serious'
    elif w > LEVELS_THRESHOLD[0] or h > LEVELS_THRESHOLD[0]:
        return 'Moderate'
    return 'Slight'


def action_judge(level):
    """判断建议行动"""
    return 'Repair' if level in ['Serious', 'Moderate'] else 'Monitor'

def img_sta(img_paths):
    """统计图片尺寸范围 [min_w, max_w, min_h, max_h]"""
    if not img_paths: return [0, 0, 0, 0]
    shape_dict = {}
    for img_path in img_paths:
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                shape_dict[os.path.basename(img_path)] = img.size
    if not shape_dict: return [0, 0, 0, 0]
    shapes = np.array(list(shape_dict.values()))
    maxs, mins = np.max(shapes, axis=0), np.min(shapes, axis=0)
    return [mins[0], maxs[0], mins[1], maxs[1]]