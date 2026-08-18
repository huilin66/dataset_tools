from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(r"\\10.22.50.44\individualdata\UAV\GPR\RadarDatas")
OUTPUT_ROOT = Path(__file__).resolve().parent / "output_images"

DTYPES = {
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}

# GPR B-scan vertical sample counts are usually hundreds or thousands.
# Keep 64 as a last resort so old data can still render, but do not prefer it.
COMMON_SAMPLES_PER_TRACE = [512, 1024, 2048, 4096, 256, 8192, 128, 64]


def summarize_binary(path: Path) -> None:
    print(f"File: {path}")
    try:
        size = path.stat().st_size
        print(f"Size: {size} bytes")

        with path.open("rb") as f:
            head = f.read(64)
        print("Head bytes:", head[:32].hex(" "))

        for dtype in DTYPES.values():
            try:
                arr = np.fromfile(path, dtype=dtype)
                if arr.size == 0:
                    continue
                print(
                    f"  dtype={dtype.__name__} -> shape={arr.shape}, "
                    f"min={arr.min():.6g}, max={arr.max():.6g}, first10={arr[:10]}"
                )
            except Exception as exc:
                print(f"  dtype={dtype.__name__} -> error: {exc}")
    except OSError as exc:
        print(f"  Unable to read file: {exc}")

    print()


def read_series(
    path: Path,
    dtype: np.dtype,
    header_bytes: int,
    complex_iq: bool,
    log_scale: bool,
) -> np.ndarray:
    with path.open("rb") as f:
        if header_bytes:
            f.seek(header_bytes)
        raw = np.fromfile(f, dtype=dtype)

    if raw.size == 0:
        raise ValueError("no data after applying header offset")

    raw = raw.astype(np.float32, copy=False)

    if complex_iq:
        if raw.size < 2:
            raise ValueError("not enough values for I/Q interpretation")
        if raw.size % 2:
            raw = raw[:-1]
        series = np.hypot(raw[0::2], raw[1::2])
    else:
        series = raw

    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if log_scale:
        min_value = float(series.min())
        if min_value < 0:
            series = series - min_value
        series = np.log1p(series)

    return series


def infer_shape(data_length: int, samples_per_trace: int | None) -> tuple[int, int]:
    if samples_per_trace is not None:
        if samples_per_trace <= 0:
            raise ValueError("--samples-per-trace must be greater than 0")
        if data_length % samples_per_trace != 0:
            raise ValueError(
                f"data length {data_length} is not divisible by "
                f"--samples-per-trace {samples_per_trace}"
            )
        return samples_per_trace, data_length // samples_per_trace

    candidates: list[tuple[float, int, int]] = []
    for rows in COMMON_SAMPLES_PER_TRACE:
        if rows >= data_length or data_length % rows != 0:
            continue
        cols = data_length // rows

        score = 0.0
        if 256 <= rows <= 4096:
            score += 100.0
        if rows < 128:
            score -= 80.0

        # Prefer a readable B-scan aspect ratio after reshape.
        aspect = cols / rows
        if 1.0 <= aspect <= 80.0:
            score += 50.0
        elif aspect > 80.0:
            score -= min(60.0, aspect / 20.0)

        score += 10.0 / (COMMON_SAMPLES_PER_TRACE.index(rows) + 1)
        candidates.append((score, rows, cols))

    if not candidates:
        raise ValueError(
            "unable to infer samples per trace; pass --samples-per-trace explicitly"
        )

    _, rows, cols = max(candidates, key=lambda item: item[0])
    return rows, cols


def normalize_to_u8(
    image: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    if not 0 <= clip_low < clip_high <= 100:
        raise ValueError("clip percentiles must satisfy 0 <= low < high <= 100")

    low, high = np.percentile(image, [clip_low, clip_high])
    if high <= low:
        low = float(image.min())
        high = float(image.max())

    image = np.clip(image, low, high)
    image = image - low
    scale = high - low
    if scale > 0:
        image = image / scale

    return (image * 255).astype(np.uint8)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    hist = np.bincount(image.ravel(), minlength=256)
    nonzero = np.flatnonzero(hist)
    if nonzero.size <= 1:
        return image

    cdf = hist.cumsum()
    cdf_min = cdf[nonzero[0]]
    total = cdf[-1]
    if total <= cdf_min:
        return image

    lookup = np.round((cdf - cdf_min) * 255 / (total - cdf_min))
    lookup = np.clip(lookup, 0, 255).astype(np.uint8)
    return lookup[image]


def downsample_for_view(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image

    step = int(np.ceil(image.shape[1] / max_width))
    trimmed_cols = (image.shape[1] // step) * step
    if trimmed_cols == 0:
        return image[:, :max_width]

    trimmed = image[:, :trimmed_cols]
    return trimmed.reshape(image.shape[0], -1, step).mean(axis=2).astype(np.uint8)


def scale_vertical(image: np.ndarray, vertical_scale: int) -> np.ndarray:
    if vertical_scale <= 1:
        return image
    return np.repeat(image, vertical_scale, axis=0)


def render_radar_file(
    path: Path,
    output_root: Path,
    dtype: np.dtype,
    header_bytes: int,
    complex_iq: bool,
    samples_per_trace: int | None,
    clip_low: float,
    clip_high: float,
    max_width: int,
    log_scale: bool,
    contrast: str,
    vertical_scale: int,
    invert: bool,
) -> None:
    print(f"Rendering: {path.name}")
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        series = read_series(path, dtype, header_bytes, complex_iq, log_scale)
        rows, cols = infer_shape(series.size, samples_per_trace)
        image = series.reshape(rows, cols)
        image_u8 = normalize_to_u8(image, clip_low, clip_high)
        if contrast == "equalize":
            image_u8 = equalize_histogram(image_u8)
        if invert:
            image_u8 = 255 - image_u8
        view_u8 = downsample_for_view(image_u8, max_width)
        view_u8 = scale_vertical(view_u8, vertical_scale)

        out_path = output_root / f"{path.stem}.png"
        Image.fromarray(view_u8, mode="L").save(out_path)
        print(
            f"  Saved {out_path} | raw shape=({rows}, {cols}), "
            f"saved shape={view_u8.shape}"
        )
        if rows <= 256:
            print(
                "  Note: vertical samples are low. If this still looks like a line, "
                "set the real --samples-per-trace from the radar metadata."
            )
    except Exception as exc:
        print(f"  Failed: {exc}")


def collect_files(data_root: Path, recursive: bool) -> list[Path]:
    try:
        iterator = data_root.rglob("*") if recursive else data_root.iterdir()
        return sorted(path for path in iterator if path.is_file())
    except PermissionError as exc:
        print(f"Cannot access data directory, permission denied: {data_root}")
        print(f"  {exc}")
    except FileNotFoundError as exc:
        print(f"Data directory does not exist: {data_root}")
        print(f"  {exc}")
    except OSError as exc:
        print(f"Cannot read data directory: {data_root}")
        print(f"  {exc}")

    return []


def parse_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read GPR binary/SLC files and export grayscale B-scan PNG images."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--extensions", default=".slc")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="int16")
    parser.add_argument("--header-bytes", type=int, default=0)
    parser.add_argument("--samples-per-trace", type=int)
    parser.add_argument("--real-only", action="store_true", help="Read one real amplitude per sample. This is the default.")
    parser.add_argument("--complex-iq", action="store_true", help="Read interleaved I/Q pairs and render their magnitude.")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.5)
    parser.add_argument("--contrast", choices=["linear", "equalize"], default="equalize")
    parser.add_argument("--vertical-scale", type=int, default=3)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--max-width", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summarize-others", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.header_bytes < 0:
        raise ValueError("--header-bytes cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if args.vertical_scale <= 0:
        raise ValueError("--vertical-scale must be greater than 0")

    extensions = parse_extensions(args.extensions)
    dtype = DTYPES[args.dtype]
    complex_iq = args.complex_iq and not args.real_only

    print("Reading GPR data from:", args.data_root)
    print("Writing images to:", args.output_root)
    print("=" * 80)

    files = collect_files(args.data_root, args.recursive)
    if not files:
        print("No files found or the directory is not accessible.")
        return

    rendered = 0
    for path in files:
        if path.suffix.lower() in extensions:
            render_radar_file(
                path=path,
                output_root=args.output_root,
                dtype=dtype,
                header_bytes=args.header_bytes,
                complex_iq=complex_iq,
                samples_per_trace=args.samples_per_trace,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
                max_width=args.max_width,
                log_scale=args.log_scale,
                contrast=args.contrast,
                vertical_scale=args.vertical_scale,
                invert=args.invert,
            )
            rendered += 1
            if args.limit is not None and rendered >= args.limit:
                break
        elif args.summarize_others:
            summarize_binary(path)

    if rendered == 0:
        print(f"No files matched extensions: {', '.join(sorted(extensions))}")


if __name__ == "__main__":
    main()
