# utils/analysis.py
import os
import numpy as np
from PIL import Image

# utils/analysis.py
import os
import numpy as np
from PIL import Image
from config import LEVELS_THRESHOLD, DISPLAY_LEVELS

def level_judge(input, unit='cm'):
    """判断缺陷等级"""
    if unit in ['pix', 'ratio']:
        xmin, ymin, xmax, ymax = input
        w = xmax - xmin
        h = ymax - ymin   
    else:
        w, h = input
    # 使用配置中的阈值
    if w > LEVELS_THRESHOLD[unit][1] or h > LEVELS_THRESHOLD[unit][1]:
        return  DISPLAY_LEVELS[2]
    elif w > LEVELS_THRESHOLD[unit][0] or h > LEVELS_THRESHOLD[unit][0]:
        return DISPLAY_LEVELS[1]
    return DISPLAY_LEVELS[0]


def action_judge_htm_t(level, category):
    if category == 'low':
        action = 'Monitor'
    elif category == 'high':
        action = 'Repair'
    elif category == 'medium':
        if level in DISPLAY_LEVELS[1:]:
            action = 'Repair'
        else:
            action = 'Monitor'
    elif category == 'leakage':
        action = 'Repair'
    else:
        action = 'Monitor'
    return action


def action_judge(level, category):
    """判断建议行动"""
    return action_judge_htm_t(level, category)

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