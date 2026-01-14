import os
from pathlib import Path

from PIL import Image
import pandas as pd

import config
from sua_bdd_tool.data.image_meta import MetadataManager
from sua_bdd_tool.utils.analysis import action_judge, level_judge
from sua_bdd_tool.utils.projection import (
    calculate_facade_gsd,
    convert_coordinate,
    pixel_to_physical,
    safe_float,
)
from sua_bdd_tool.utils.visualization import crop_box, draw_box


class ImageProcessor:
    def __init__(self, labels, colors, vis_dir, crop_dir, vis_aux_dir=None, crop_aux_dir=None, metadata_provider=None, exif_db=None):
        self.meta_mgr = MetadataManager()
        self.labels = labels
        self.colors = colors
        self.vis_dir = vis_dir
        self.vis_aux_dir = vis_aux_dir
        self.crop_dir = crop_dir
        self.crop_aux_dir = crop_aux_dir
        self.metadata_provider = metadata_provider
        self.exif_db = exif_db

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
        
        gsd = calculate_facade_gsd(dist_mm, focal, specs['sensor_width_mm'], img_w)

        # 可视化
        vis_path = os.path.join(self.vis_dir, f"{stem_name}.png")
        draw_box(img.copy(), detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, stem_name)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, detections)

        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            w_cm = pixel_to_physical(bbox[4]-bbox[2], gsd)
            h_cm = pixel_to_physical(bbox[5]-bbox[3], gsd)


            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)

            crop_p = os.path.join(crop_subdir, f"{i}.png")
            crops[i].save(crop_p)

            res = {
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
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

class DedupProcessor(ImageProcessor):
    """
    专用处理器：负责图像处理 + 提取原始 GPS 信息
    """
    def process(self, item):
        img_path = item['image_path']
        detections = item['detections']
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem
        
        img = Image.open(img_path).convert('RGB')

        img_exif = self.exif_db.get(img_name, None)
        focal_length = img_exif.get('focal', None) if img_exif else None
        lrf_dist = img_exif.get('lrf_dist', None) if img_exif else None
        lon = img_exif.get('lon', None) if img_exif else None   
        lat = img_exif.get('lat', None) if img_exif else None   
        gps_str = convert_coordinate(lat, lon)

        if img_exif["model"] in ["DJI M4T", "M4T"]:
            if img_exif["camera_type"] in ["WideCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Wide"]["pixel_size_um"]
            elif img_exif["camera_type"] in ["InfraredCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Thermal"]["pixel_size_um"]
            else:
                raise ValueError(f"Unknown camera_type: {img_exif['camera_type']}")
        
        gsd = calculate_facade_gsd(lrf_dist, focal_length, pixel_size_um)


        # 2. 可视化
        vis_path = os.path.join(self.vis_dir, f"{img_stem}.png")
        vis_detections = detections[:, :] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        
        crop_subdir = os.path.join(self.crop_dir, img_stem)
        os.makedirs(crop_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)

        # 3. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            id = int(bbox[6]) # Dedup ID
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)

            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)
            
            crop_path = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_path)

            res = {
                'Path': img_path,
                'VisPath': vis_path, 
                'CropPath': crop_path, 
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                'W_cm': float(f"{w_cm:.2f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.2f}") if h_cm else "N/A", 
                'Area_cm2': w_cm*h_cm if (w_cm and h_cm) else "N/A",
                
                # === 注入 GPS ===
                'GPS': gps_str,
                
                # 内部字段
                'id': id,
                'img_name': img_name
            }
            records.append(res)
            
        return pd.DataFrame(records)




class DedupProcessor_AuxImage(ImageProcessor):
    """
    专用处理器：负责图像处理 + 提取原始 GPS 信息
    """
    def process(self, item):
        detections = item['detections']
        img_path = item['image_path']
        img_aux_path = item['image_aux_path']
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem

        img = Image.open(img_path).convert('RGB')
        img_aux = Image.open(img_aux_path).convert('RGB')

        img_exif = self.exif_db.get(img_name, None)
        focal_length = img_exif.get('focal', None) if img_exif else None
        lrf_dist = img_exif.get('lrf_dist', None) if img_exif else None
        lon = img_exif.get('lon', None) if img_exif else None   
        lat = img_exif.get('lat', None) if img_exif else None   
        gps_str = convert_coordinate(lat, lon)

        if img_exif["model"] in ["DJI M4T", "M4T"]:
            if img_exif["camera_type"] in ["WideCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Wide"]["pixel_size_um"]
            elif img_exif["camera_type"] in ["InfraredCamera"]:
                pixel_size_um = config.CAMERA_PARAMS["M4T_Thermal"]["pixel_size_um"]
            else:
                raise ValueError(f"Unknown camera_type: {img_exif['camera_type']}")
        
        gsd = calculate_facade_gsd(lrf_dist, focal_length, pixel_size_um)


        # 2. 可视化
        vis_path = os.path.join(self.vis_dir, f"{img_stem}.png")
        vis_aux_path = os.path.join(self.vis_aux_dir, f"{img_stem}.png")

        vis_detections = detections[:, :] if len(detections) > 0 else []
        draw_box(img.copy(), vis_detections, self.labels, self.colors).save(vis_path)
        draw_box(img_aux.copy(), vis_detections, self.labels, self.colors).save(vis_aux_path)
        
        crop_subdir = os.path.join(self.crop_dir, img_stem)
        crop_aux_subdir = os.path.join(self.crop_aux_dir, img_stem)
        os.makedirs(crop_subdir, exist_ok=True)
        os.makedirs(crop_aux_subdir, exist_ok=True)
        crops = crop_box(img, vis_detections)
        crops_aux = crop_box(img_aux, vis_detections)

        # 3. 生成记录
        records = []
        for i, bbox in enumerate(detections):
            cls_id = int(bbox[0])
            id = int(bbox[6]) # Dedup ID
            
            w_pix = int(bbox[4]-bbox[2])
            h_pix = int(bbox[5]-bbox[3])
            w_cm = pixel_to_physical(w_pix, gsd)
            h_cm = pixel_to_physical(h_pix, gsd)

            category = self.labels[cls_id] if cls_id < len(self.labels) else f"Class_{cls_id}"
            level = level_judge([w_cm, h_cm])
            action = action_judge(level, category)
            
            crop_path = os.path.join(crop_subdir, f"{i}.png")
            if i < len(crops): crops[i].save(crop_path)
            crop_aux_path = os.path.join(crop_aux_subdir, f"{i}.png")
            if i < len(crops_aux): crops_aux[i].save(crop_aux_path)
            
            res = {
                'Path': img_path,
                'VisPath': vis_path, 
                'VisAuxPath': vis_aux_path,
                'CropPath': crop_path, 
                'CropAuxPath': crop_aux_path,
                'Category': category,
                'Level': level,
                'Score': float(bbox[1]),
                'Action': action,
                'W_pix': w_pix, 'H_pix': h_pix, 'Area_pix': w_pix * h_pix,
                'W_cm': float(f"{w_cm:.2f}") if w_cm else "N/A",
                'H_cm': float(f"{h_cm:.2f}") if h_cm else "N/A", 
                'Area_cm2': w_cm*h_cm if (w_cm and h_cm) else "N/A",
                
                # === 注入 GPS ===
                'GPS': gps_str,
                
                # 内部字段
                'id': id,
                'img_name': img_name
            }
            records.append(res)
            
        return pd.DataFrame(records)


