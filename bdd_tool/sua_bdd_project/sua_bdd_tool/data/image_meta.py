
from calendar import c
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm

import config
from sua_bdd_tool.utils.projection import (
    get_cardinal_direction,
    get_exif,
    parse_float,
    parse_gps_from_exif,
    parse_dji_xmp,
)

@dataclass
class ImageMeta:
    """
    定义每张图片的元数据结构
    """
    filename: str
    rel_dir: str
    root_dir: str
    
    lat: Optional[float] = None
    lon: Optional[float] = None
    abs_alt: Optional[float] = None
    rel_alt: Optional[float] = None

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    direction: Optional[str] = None

    fov: Optional[float] = None
    lrf_status: Optional[str] = None
    lrf_lat: Optional[float] = None
    lrf_lon: Optional[float] = None
    lrf_dist: Optional[float] = None

    fov: Optional[float] = None
    focal_35: Optional[float] = None
    zoom_ratio: Optional[float] = None
    camera_type: Optional[str] = None
    capture_time: Optional[str] = None
    shutter_type: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

# TODO remove this class in the future
class MetadataManager:
    '''
    over version class, will be remove in the future
    '''
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

def process_single_image(file_path, root_dir):
    """
    处理单张图片：读取 -> 解析 -> 返回 ImageMeta 对象
    """
    exif = get_exif(file_path)
    rel_dir = str(file_path.relative_to(root_dir).parent)

    lat, lon = parse_gps_from_exif(exif)
    abs_alt = parse_float(exif.get('AbsoluteAltitude') or exif.get('GPSAltitude'))
    rel_alt = parse_float(exif.get('RelativeAltitude'))

    yaw = (parse_float(exif.get("GimbalYawDegree") or exif.get("FlightYawDegree") or 0)) % 360
    pitch = parse_float(exif.get("GimbalPitchDegree") or exif.get("FlightPitchDegree") or 0)
    roll = parse_float(exif.get("GimbalRollDegree") or exif.get("FlightRollDegree") or 0)

    direction = get_cardinal_direction(yaw)

    lfr_status = exif.get('LRFStatus')
    lrf_lat = parse_float(exif.get("LRFTargetLat")) if lfr_status == 'Normal' else None
    lrf_lon = parse_float(exif.get("LRFTargetLon")) if lfr_status == 'Normal' else None
    lrf_dist = parse_float(exif.get("LRFTargetDistance")) if lfr_status == 'Normal' else None

    fov = parse_float(exif.get("FOV"))
    focal_35 = parse_float(exif.get('FocalLengthIn35mmFormat') or exif.get('FocalLength35efl'))
    zoom_ratio = parse_float(exif.get('DigitalZoomRatio'))
    if zoom_ratio != 1:
        print(zoom_ratio, file_path)
    camera_type = exif.get('ImageSource')
    capture_time = exif.get('DateTimeOriginal')
    shutter_type = exif.get('ShutterType') or 'Unknown'
    model = exif.get('UniqueCameraModel') or exif.get('Model')

    return ImageMeta(
        filename=file_path.name,
        rel_dir=rel_dir,
        root_dir=str(root_dir),
        lat=lat,
        lon=lon,
        abs_alt=abs_alt,
        rel_alt=rel_alt,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        direction=direction,
        lrf_status=lfr_status,
        lrf_lat=lrf_lat,
        lrf_lon=lrf_lon,
        lrf_dist=lrf_dist,
        fov=fov,
        focal_35=focal_35,
        zoom_ratio=zoom_ratio,
        camera_type=camera_type,
        capture_time=capture_time,
        shutter_type=shutter_type,
        model=model,
    )

# ================= 主流程 =================

def build_metadata_json(source_root, json_save_path, num_workers=1):
    root = Path(source_root)
    if not root.exists():
        print(f"错误: 路径不存在 {source_root}")
        return

    # 1. 递归扫描多级文件夹 (Recursive Scan)
    print(f"正在递归扫描 '{source_root}' ...")
    
    # 定义支持的图片后缀 (忽略大小写)
    valid_exts = {'.jpg', '.jpeg', '.png', '.tiff'}
    
    # rglob('*') 会遍历所有子目录
    all_files = [
        p for p in root.rglob('*') 
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    
    total_files = len(all_files)
    print(f"共找到 {total_files} 张图片。")
    print(f"模式: {'多线程并行 (' + str(num_workers) + ' workers)' if num_workers > 1 else '单线程顺序执行'}")

    metadata_list = []

    # 2. 执行提取 (单线程/多线程分支)
    if num_workers > 1:
        # --- 多线程模式 ---
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 使用 lambda 绑定 root_dir 参数
            # tqdm 显示总体进度
            results = list(tqdm(
                executor.map(lambda p: process_single_image(p, root), all_files),
                total=total_files,
                desc="Extracting EXIF (Multi)"
            ))
            metadata_list = results
    else:
        # --- 单线程模式 (调试推荐) ---
        for p in tqdm(all_files, desc="Extracting EXIF (Single)"):
            res = process_single_image(p, root)
            metadata_list.append(res)

    # 3. 构建字典并保存
    # 按照你的要求，Key = 文件名
    json_db = {}
    
    for meta in metadata_list:
        if meta is not None:
            # 如果出现重名文件，这里会发生覆盖，请知悉
            json_db[meta.filename] = meta.to_dict()

    print(f"解析成功: {len(json_db)} / {total_files} (如有重名文件已被覆盖)")
    print(f"正在保存到 {json_save_path} ...")
    
    # 确保输出目录存在
    Path(json_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(json_db, f, indent=2, ensure_ascii=False)
    
    print("完成！")

