import re
from pathlib import Path

import cv2

# ==================== 配置 ====================

# 图片所在文件夹
image_folder = Path(r"\\158.132.186.40\isds\huilin\tp\traffic sign\track2")

# 输出视频路径
output_path = Path(r"E:\output2.mp4")

# 视频帧率
fps = 25

# 支持的图片格式
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ==================== 自然排序 ====================


def natural_sort_key(path: Path):
    """
    按照类似 Windows 文件夹“名称排序”的方式排序。

    例如：
    image_1.png
    image_2.png
    image_10.png
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


# ==================== 读取并排序 ====================

if not image_folder.exists():
    raise FileNotFoundError(f"图片文件夹不存在：{image_folder}")

image_paths = [
    path
    for path in image_folder.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
]

image_paths.sort(key=natural_sort_key)

if not image_paths:
    raise RuntimeError(f"文件夹中没有找到支持的图片：{image_folder}")

print(f"共找到 {len(image_paths)} 张图片：")

for index, path in enumerate(image_paths, start=1):
    print(f"{index:04d}: {path.name}")


# ==================== 读取第一张图片 ====================

first_frame = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)

if first_frame is None:
    raise RuntimeError(f"无法读取第一张图片：{image_paths[0]}")

height, width = first_frame.shape[:2]

print(f"视频分辨率：{width} × {height}")
print(f"视频帧率：{fps} FPS")


# ==================== 创建视频 ====================

output_path.parent.mkdir(parents=True, exist_ok=True)

video_writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not video_writer.isOpened():
    raise RuntimeError(f"无法创建视频文件：{output_path}")


# ==================== 写入每一帧 ====================

written_count = 0
skipped_count = 0

for index, image_path in enumerate(image_paths, start=1):
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        print(f"[跳过] 无法读取：{image_path.name}")
        skipped_count += 1
        continue

    # 如果图片尺寸与第一张不一致，统一调整尺寸
    if frame.shape[:2] != (height, width):
        print(
            f"[调整尺寸] {image_path.name}: "
            f"{frame.shape[1]}×{frame.shape[0]} → {width}×{height}"
        )

        frame = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    video_writer.write(frame)
    written_count += 1

    print(
        f"\r正在写入：{index}/{len(image_paths)}",
        end="",
        flush=True,
    )


# ==================== 完成 ====================

video_writer.release()

print("\n")
print("视频生成完成")
print(f"输出路径：{output_path}")
print(f"成功写入：{written_count} 帧")
print(f"跳过图片：{skipped_count} 张")
print(f"视频时长：{written_count / fps:.2f} 秒")
