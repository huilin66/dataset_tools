# utils/metadata.py
import re
import os
from PIL import Image
from utils.exif_dji_xmp import parse_dji_xmp
import config

def safe_float(value):
    """
    处理 '6.7 mm', '4.67 m', None, 0 等各种情况转 float
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    
    val_str = str(value).strip()
    try:
        return float(val_str)
    except ValueError:
        # 提取数字部分 (支持 "6.7 mm", "+93.9 m")
        match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
        if match:
            return float(match.group())
        return 0.0

class MetadataManager:
    def __init__(self):
        self.method_used = "Unknown"

    def get_standard_exif(self, img):
        """[Fallback] 使用 PIL 获取标准 EXIF"""
        exif_data = {}
        try:
            info = img.getexif()
            if info:
                if 272 in info: exif_data['Model'] = str(info[272]).strip()
                if 34665 in info:
                    sub_ifd = info.get_ifd(34665)
                    if 37386 in sub_ifd:
                        exif_data['FocalLength'] = float(sub_ifd[37386])
        except Exception:
            pass
        return exif_data

    def get_unified_metadata(self, img_path, img_pil):
        """双保险元数据获取策略"""
        meta = {}
        try:
            import pyexif
            exif_editor = pyexif.ExifEditor(str(img_path))
            meta = exif_editor.getDictTags()
            meta['_parsing_method'] = "PyExif (ExifTool)"
        except (ImportError, FileNotFoundError, Exception):
            meta.update(self.get_standard_exif(img_pil))
            meta.update(parse_dji_xmp(img_path))
            meta['_parsing_method'] = "Fallback (PIL + XMP)"

        if 'Model' not in meta and 'DroneModel' in meta:
            meta['Model'] = meta['DroneModel']
        return meta

    def get_camera_specs(self, meta_dict, filename):
        """匹配硬件参数库"""
        model = meta_dict.get('Model') or meta_dict.get('DroneModel') or 'Unknown'
        if 'Matrice 4' in str(model) or 'M4' in str(model): model = 'M4T'
        elif 'Mavic 3 Thermal' in str(model): model = 'M3T'
        elif 'Mavic 3 Enterprise' in str(model): model = 'M3E'
        elif 'Matrice 30' in str(model): model = 'M30T'
        
        img_source = str(meta_dict.get('ImageSource', ''))
        is_thermal = '_T' in filename or 'Thermal' in img_source or 'IR' in img_source
        cam_type = 'Thermal' if is_thermal else 'Wide'
        
        config_key = f"{model}_{cam_type}"
        specs = config.DRONE_PARAMS.get(config_key, config.DRONE_PARAMS['default'])
        return specs, model