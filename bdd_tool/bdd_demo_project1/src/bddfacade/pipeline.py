from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .exif_dji_xmp import parse_dji_xmp
from .rgbt_indexer import ImageItem, build_index, save_index


def _f(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x.replace("+", ""))
    except Exception:
        return None


@dataclass
class PoseRecord:
    image_id: str
    rgb_path: str
    t_path: Optional[str]

    # RTK/定位
    gps_status: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    abs_alt: Optional[float]
    rel_alt: Optional[float]

    # 云台姿态（建议用于相机朝向）
    gimbal_yaw: Optional[float]
    gimbal_pitch: Optional[float]
    gimbal_roll: Optional[float]

    # 飞行器姿态（可选）
    flight_yaw: Optional[float]
    flight_pitch: Optional[float]
    flight_roll: Optional[float]

    # RTK质量
    rtk_std_lat: Optional[float]
    rtk_std_lon: Optional[float]
    rtk_std_hgt: Optional[float]
    rtk_diff_age: Optional[float]
    rtk_flag: Optional[str]

    # 触发时间（用于跨目录配对或对齐）
    utc_at_exposure: Optional[str]

    # LRF（可选，用于尺度/锚点）
    lrf_status: Optional[str]
    lrf_distance: Optional[float]
    lrf_target_lat: Optional[float]
    lrf_target_lon: Optional[float]
    lrf_target_alt: Optional[float]
    lrf_target_abs_alt: Optional[float]

    # 其他
    image_source: Optional[str]
    product_name: Optional[str]
    drone_model: Optional[str]


def extract_pose_for_item(item: ImageItem) -> PoseRecord:
    meta = parse_dji_xmp(item.rgb_path)

    return PoseRecord(
        image_id=item.id,
        rgb_path=item.rgb_path,
        t_path=item.t_path,

        gps_status=meta.get("GpsStatus"),
        lat=_f(meta.get("GpsLatitude")),
        lon=_f(meta.get("GpsLongitude")),
        abs_alt=_f(meta.get("AbsoluteAltitude")),
        rel_alt=_f(meta.get("RelativeAltitude")),

        gimbal_roll=_f(meta.get("GimbalRollDegree")),
        gimbal_yaw=_f(meta.get("GimbalYawDegree")),
        gimbal_pitch=_f(meta.get("GimbalPitchDegree")),

        flight_roll=_f(meta.get("FlightRollDegree")),
        flight_yaw=_f(meta.get("FlightYawDegree")),
        flight_pitch=_f(meta.get("FlightPitchDegree")),

        rtk_std_lon=_f(meta.get("RtkStdLon")),
        rtk_std_lat=_f(meta.get("RtkStdLat")),
        rtk_std_hgt=_f(meta.get("RtkStdHgt")),
        rtk_diff_age=_f(meta.get("RtkDiffAge")),
        rtk_flag=meta.get("RtkFlag"),

        utc_at_exposure=meta.get("UTCAtExposure"),

        lrf_status=meta.get("LRFStatus"),
        lrf_distance=_f(meta.get("LRFTargetDistance")),
        lrf_target_lon=_f(meta.get("LRFTargetLon")),
        lrf_target_lat=_f(meta.get("LRFTargetLat")),
        lrf_target_alt=_f(meta.get("LRFTargetAlt")),
        lrf_target_abs_alt=_f(meta.get("LRFTargetAbsAlt")),

        image_source=meta.get("ImageSource"),
        product_name=meta.get("ProductName"),
        drone_model=meta.get("DroneModel"),
    )


def build_poses(items: List[ImageItem]) -> List[Dict[str, Any]]:
    poses: List[Dict[str, Any]] = []
    for it in tqdm(items, desc="Extract DJI pose"):
        poses.append(asdict(extract_pose_for_item(it)))
    print('pose info:')
    print(f'total {len(poses)} poses')
    return poses


def save_json(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'save {out_path}')


if __name__ == "__main__":
    pass
    data_root = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\collected data'
    out_dir = r'outputs'

    data_root = Path(data_root)
    out_dir = Path(out_dir)

    items = build_index(data_root)
    save_index(items, out_dir / "index.json")

    poses = build_poses(items)
    save_json(poses, out_dir / "poses_rgb.json")

    # pairs：基于 index 直接输出
    pairs = [{"rgb": x.rgb_path, "t": x.t_path} for x in items if x.t_path]
    save_json(pairs, out_dir / "pairs.json")

    print(f"\nDone. index={out_dir/'index.json'} poses={out_dir/'poses_rgb.json'} pairs={out_dir/'pairs.json'}")

    #  C:/Users/USER/.conda/envs/common/python.exe -m bddfacade.server --host 127.0.0.1 --port 8080