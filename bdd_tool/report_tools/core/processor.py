# core/processor.py
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from utils.metadata import MetadataManager, safe_float
from utils.geo_utils import calculate_gsd, pixel_to_physical
from utils.visualization import draw_box, crop_box
from utils.analysis import level_judge, action_judge

class ImageProcessor:
    def __init__(self, labels, colors, vis_dir, crop_dir, metadata_provider=None):
        self.meta_mgr = MetadataManager()
        self.labels = labels
        self.colors = colors
        self.vis_dir = vis_dir
        self.crop_dir = crop_dir
        self.metadata_provider = metadata_provider

    def process(self, item):
        img_path = item['image_path']
        detections = item['detections']
        stem_name = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        all_meta = self.meta_mgr.get_unified_metadata(img_path, img)

        # 硬件与GSD计算
        specs, _ = self.meta_mgr.get_camera_specs(all_meta, stem_name)
        focal = safe_float(all_meta.get('FocalLength')) or specs['focal_length_mm']
        
        dist_mm = safe_float(all_meta.get('LRFTargetDistance')) * 1000
        if dist_mm == 0:
            dist_mm = abs(safe_float(all_meta.get('RelativeAltitude'))) * 1000
        if dist_mm == 0:
            import config
            dist_mm = getattr(config, 'DEFAULT_DISTANCE_M', 15.0) * 1000
        
        gsd = calculate_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 可视化
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        draw_box(img.copy(), detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, detections)

        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            level = level_judge(bbox[2:])
            w_cm = pixel_to_physical(bbox[4]-bbox[2], gsd)
            h_cm = pixel_to_physical(bbox[5]-bbox[3], gsd)
            
            crop_p = os.path.join(crop_subdir, f"{i}.png")
            crops[i].save(crop_p)

            res = {
                'Category': self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}",
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action_judge(level),
                'W_pix': int(bbox[4]-bbox[2]),
                'H_pix': int(bbox[5]-bbox[3]),
                'Area_pix': int((bbox[4]-bbox[2]) * (bbox[5]-bbox[3])),
                'W_cm': f"{w_cm:.1f}" if w_cm else "N/A",
                'H_cm': f"{h_cm:.1f}" if h_cm else "N/A",
                'Area_cm2': f"{(w_cm * h_cm):.1f}" if (w_cm and h_cm) else "N/A",
                'VisPath': vis_path, 'CropPath': crop_p, 'Path': img_path
            }
            if self.metadata_provider:
                ext_meta = self.metadata_provider(img_path)
                if ext_meta: res.update(ext_meta)
            records.append(res)
            
        return pd.DataFrame(records)