# loaders/yolo_loader.py
import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image

class YoloLoader:
    def __init__(self, img_dir, txt_dir, class_list=None):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.class_list = class_list if class_list else []

    def _yolo_norm_to_pixel(self, yolo_line, img_w, img_h):
        """解析单行 YOLO 格式"""
        parts = yolo_line.strip().split()
        cls_id = int(parts[0])
        
        if len(parts) >= 6:
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
        else:
            xc, yc, w, h = map(float, parts[1:5])
            conf = 1.0

        x_center = xc * img_w
        y_center = yc * img_h
        width = w * img_w
        height = h * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [cls_id, conf, x1, y1, x2, y2]

    def load(self):
        """
        加载数据并返回标准格式列表
        Returns: List of dict {'image_path': str, 'detections': np.array}
        """
        data_list = []
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext.upper())))
        
        img_paths = sorted(list(set(img_paths))) 
        print(f"[Loader] Found {len(img_paths)} images in {self.img_dir}")

        for img_path in img_paths:
            stem = Path(img_path).stem
            txt_path = os.path.join(self.txt_dir, stem + '.txt')
            detections = []
            
            with Image.open(img_path) as img:
                w, h = img.size
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            det = self._yolo_norm_to_pixel(line, w, h)
                            detections.append(det)
            
            data_list.append({
                'image_path': img_path,
                'detections': np.array(detections) if detections else np.array([])
            })
            
        return data_list