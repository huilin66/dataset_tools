#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

# 可选：支持 iPhone/Android 的 HEIC、HEIF 文件
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".heic",
    ".heif",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
}


def rational_to_float(value: Any) -> float:
    """将 EXIF Rational、元组或普通数值转换为 float。"""
    if value is None:
        raise ValueError("数值为空")

    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        if denominator == 0:
            raise ZeroDivisionError("EXIF 分母为 0")
        return float(numerator) / float(denominator)

    return float(value)


def dms_to_decimal(dms: Any, direction: str) -> float:
    """
    将度分秒坐标转换为十进制度数。

    例如：
        (22°, 18', 30") -> 22.308333
    """
    if not dms or len(dms) != 3:
        raise ValueError(f"无效的度分秒坐标：{dms}")

    degrees = rational_to_float(dms[0])
    minutes = rational_to_float(dms[1])
    seconds = rational_to_float(dms[2])

    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    direction = str(direction).strip().upper()
    if direction in {"S", "W"}:
        decimal = -decimal

    return decimal


def decode_exif_text(value: Any) -> str:
    """将 EXIF 字节或普通值转换为字符串。"""
    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00 ")

    return str(value).strip()


def get_exif_data(image: Image.Image) -> dict[str, Any]:
    """获取普通 EXIF 信息，并将标签编号转换为标签名称。"""
    exif = image.getexif()
    if not exif:
        return {}

    return {
        ExifTags.TAGS.get(tag_id, str(tag_id)): value for tag_id, value in exif.items()
    }


def get_gps_data(image: Image.Image) -> dict[str, Any]:
    """获取 GPS EXIF 信息，并转换 GPS 标签名称。"""
    exif = image.getexif()
    if not exif:
        return {}

    gps_ifd = {}

    # 新版 Pillow 的读取方式
    try:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (AttributeError, KeyError, TypeError):
        pass

    # 兼容旧版 Pillow
    if not gps_ifd:
        gps_tag_id = next(
            (
                tag_id
                for tag_id, tag_name in ExifTags.TAGS.items()
                if tag_name == "GPSInfo"
            ),
            None,
        )

        if gps_tag_id is not None:
            raw_gps = exif.get(gps_tag_id)
            if isinstance(raw_gps, dict):
                gps_ifd = raw_gps

    return {
        ExifTags.GPSTAGS.get(tag_id, str(tag_id)): value
        for tag_id, value in gps_ifd.items()
    }


def extract_photo_info(photo_path: Path) -> dict[str, Any]:
    """提取单张照片的坐标及其他 EXIF 信息。"""
    result = {
        "filename": photo_path.name,
        "relative_path": "",
        "latitude": "",
        "longitude": "",
        "altitude_m": "",
        "datetime_original": "",
        "make": "",
        "model": "",
        "map_url": "",
        "status": "",
        "error": "",
    }

    try:
        with Image.open(photo_path) as image:
            exif_data = get_exif_data(image)
            gps_data = get_gps_data(image)

            result["datetime_original"] = decode_exif_text(
                exif_data.get("DateTimeOriginal")
                or exif_data.get("DateTimeDigitized")
                or exif_data.get("DateTime")
            )
            result["make"] = decode_exif_text(exif_data.get("Make"))
            result["model"] = decode_exif_text(exif_data.get("Model"))

            latitude_dms = gps_data.get("GPSLatitude")
            latitude_ref = decode_exif_text(gps_data.get("GPSLatitudeRef"))

            longitude_dms = gps_data.get("GPSLongitude")
            longitude_ref = decode_exif_text(gps_data.get("GPSLongitudeRef"))

            if latitude_dms and longitude_dms:
                latitude = dms_to_decimal(latitude_dms, latitude_ref)
                longitude = dms_to_decimal(longitude_dms, longitude_ref)

                result["latitude"] = f"{latitude:.8f}"
                result["longitude"] = f"{longitude:.8f}"
                result["map_url"] = (
                    f"https://www.google.com/maps?q={latitude:.8f},{longitude:.8f}"
                )

                altitude = gps_data.get("GPSAltitude")
                altitude_ref = gps_data.get("GPSAltitudeRef", 0)

                if altitude is not None:
                    altitude_value = rational_to_float(altitude)

                    # GPSAltitudeRef = 1 表示低于海平面
                    try:
                        altitude_ref_value = int(altitude_ref)
                    except (TypeError, ValueError):
                        altitude_ref_value = 0

                    if altitude_ref_value == 1:
                        altitude_value = -altitude_value

                    result["altitude_m"] = f"{altitude_value:.2f}"

                result["status"] = "GPS found"
            else:
                result["status"] = "No GPS"
                result["error"] = "照片中没有 GPS EXIF 信息"

    except Exception as exc:
        result["status"] = "Read failed"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def find_photos(input_dir: Path, recursive: bool = True) -> list[Path]:
    """查找目录中的所有支持格式照片。"""
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量提取手机照片 EXIF 坐标，并导出 CSV。"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="照片所在目录",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("photo_gps.csv"),
        help="输出 CSV 路径，默认：photo_gps.csv",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="只处理当前目录，不递归扫描子目录",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录：{input_dir}")

    photos = find_photos(
        input_dir=input_dir,
        recursive=not args.no_recursive,
    )

    if not photos:
        print(f"未找到支持的照片文件：{input_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "relative_path",
        "latitude",
        "longitude",
        "altitude_m",
        "datetime_original",
        "make",
        "model",
        "map_url",
        "status",
        "error",
    ]

    gps_count = 0
    no_gps_count = 0
    failed_count = 0

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, photo_path in enumerate(photos, start=1):
            result = extract_photo_info(photo_path)
            result["relative_path"] = str(photo_path.relative_to(input_dir))

            writer.writerow(result)

            if result["status"] == "GPS found":
                gps_count += 1
            elif result["status"] == "No GPS":
                no_gps_count += 1
            else:
                failed_count += 1

            print(
                f"[{index}/{len(photos)}] {result['status']}: {result['relative_path']}"
            )

    print("\n处理完成")
    print(f"照片总数：{len(photos)}")
    print(f"包含坐标：{gps_count}")
    print(f"无坐标：{no_gps_count}")
    print(f"读取失败：{failed_count}")
    print(f"结果文件：{output_path}")

    if not HEIF_SUPPORTED:
        print("\n提示：当前未安装 pillow-heif，如果目录中包含 HEIC/HEIF 文件，请执行：")
        print("pip install pillow-heif")


if __name__ == "__main__":
    main()
