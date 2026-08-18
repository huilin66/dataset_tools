"""Extract images from an Insta360 ``.insv`` video.

The actual decoding is delegated to the FFmpeg backend bundled with OpenCV.
This keeps the extractor streaming: only one decoded frame is held in memory
at a time, which is important for high-resolution 360-degree videos.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union


PathLike = Union[str, Path]
SUPPORTED_FORMATS = ("jpg", "jpeg", "png")


@dataclass(frozen=True)
class ExtractionResult:
    """Summary returned after a frame extraction run."""

    input_path: Path
    output_dir: Path
    frames_read: int
    frames_saved: int
    frames_skipped: int
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    reported_frame_count: Optional[int]


def _load_cv2():
    """Import OpenCV lazily so ``--help`` remains available without it."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "未安装 OpenCV，请先执行: python -m pip install -r "
            "insta_pan_tools/requirements.txt"
        ) from exc
    return cv2


def _load_tqdm():
    """Import tqdm lazily so the module can still display CLI help."""

    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "未安装 tqdm，请先执行: python -m pip install -r "
            "insta_pan_tools/requirements.txt"
        ) from exc
    return tqdm


def _normalise_format(image_format: str) -> str:
    extension = image_format.lower().lstrip(".")
    if extension not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(f"不支持的图片格式: {image_format!r}，可选格式: {supported}")
    return "jpg" if extension == "jpeg" else extension


def _optional_int(value: float) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return int(round(value))


def _optional_float(value: float) -> Optional[float]:
    if value is None or value <= 0:
        return None
    return float(value)


def extract_frames(
    input_path: PathLike,
    output_dir: Optional[PathLike] = None,
    *,
    image_format: str = "png",
    frame_step: int = 1,
    jpeg_quality: int = 95,
    overwrite: bool = False,
    show_progress: bool = True,
) -> ExtractionResult:
    """Extract frames from an Insta360 ``.insv`` file.

    Parameters
    ----------
    input_path:
        Path to the source ``.insv`` file.
    output_dir:
        Directory for extracted images. By default, ``<video>_frames`` is
        created next to the source video.
    image_format:
        ``jpg``/``jpeg`` or ``png``. PNG is the default because it is lossless;
        use JPEG when a smaller output directory is preferred.
    frame_step:
        Save one frame every ``frame_step`` decoded frames. The default of 1
        saves every frame.
    jpeg_quality:
        JPEG quality from 0 to 100. It is ignored for PNG output.
    overwrite:
        Overwrite an existing output image with the same frame number. Without
        this flag, existing images are left untouched.
    show_progress:
        Show a terminal progress bar while decoding. Set to ``False`` when the
        function is called from a non-interactive application.

    Returns
    -------
    ExtractionResult
        Counts and basic stream metadata for the extraction.
    """

    source = Path(input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"找不到输入文件: {source}")
    if not source.is_file():
        raise ValueError(f"输入路径不是文件: {source}")
    if source.suffix.lower() != ".insv":
        raise ValueError(f"输入文件必须是 .insv 文件: {source}")
    if frame_step < 1:
        raise ValueError("frame_step 必须大于或等于 1")
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality 必须在 0 到 100 之间")

    extension = _normalise_format(image_format)
    destination = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else source.with_name(f"{source.stem}_frames")
    )
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"输出路径不是目录: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    cv2 = _load_cv2()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "无法打开 .insv 文件。请确认 OpenCV 带有 FFmpeg 解码支持，"
            "且视频文件未损坏。"
        )

    reported_frame_count = _optional_int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    fps = _optional_float(capture.get(cv2.CAP_PROP_FPS))
    width = _optional_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _optional_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_read = 0
    frames_saved = 0
    frames_skipped = 0
    tqdm = _load_tqdm()
    progress = tqdm(
        total=reported_frame_count,
        desc="提取帧",
        unit="帧",
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    write_params = (
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        if extension == "jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index = frames_read
            frames_read += 1
            progress.update(1)
            if frame_index % frame_step != 0:
                continue

            output_path = destination / f"frame_{frame_index:06d}.{extension}"
            if output_path.exists() and not overwrite:
                frames_skipped += 1
                continue

            if not cv2.imwrite(str(output_path), frame, write_params):
                raise OSError(f"写入图片失败: {output_path}")
            frames_saved += 1
    finally:
        capture.release()
        progress.close()

    return ExtractionResult(
        input_path=source,
        output_dir=destination,
        frames_read=frames_read,
        frames_saved=frames_saved,
        frames_skipped=frames_skipped,
        fps=fps,
        width=width,
        height=height,
        reported_frame_count=reported_frame_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Insta360 .insv 视频逐帧提取图片。"
    )
    parser.add_argument("input", type=Path, help="输入 .insv 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出目录，默认是输入文件旁的 <文件名>_frames",
    )
    parser.add_argument(
        "--format",
        dest="image_format",
        choices=SUPPORTED_FORMATS,
        default="png",
        help="输出图片格式，默认 png（无损）",
    )
    parser.add_argument(
        "--every",
        dest="frame_step",
        type=int,
        default=1,
        metavar="N",
        help="每隔 N 帧保存一帧；默认 1，即保存全部帧",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        metavar="0-100",
        help="JPEG 质量，默认 95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出目录中同名的已存在图片",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示终端进度条",
    )
    return parser


def _format_metadata(result: ExtractionResult) -> str:
    size = (
        f"{result.width}x{result.height}"
        if result.width is not None and result.height is not None
        else "未知"
    )
    fps = f"{result.fps:.3f}" if result.fps is not None else "未知"
    reported = (
        str(result.reported_frame_count)
        if result.reported_frame_count is not None
        else "未知"
    )
    return f"分辨率: {size}，FPS: {fps}，文件报告帧数: {reported}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = extract_frames(
            args.input,
            args.output,
            image_format=args.image_format,
            frame_step=args.frame_step,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"输入: {result.input_path}")
    print(f"输出目录: {result.output_dir}")
    print(_format_metadata(result))
    print(
        f"完成：读取 {result.frames_read} 帧，保存 {result.frames_saved} 帧，"
        f"跳过 {result.frames_skipped} 帧。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
