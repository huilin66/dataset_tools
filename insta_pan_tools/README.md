# Insta360 `.insv` 帧提取工具

给定 Insta360 的 `.insv` 视频，使用 OpenCV/FFmpeg 逐帧解码并保存为图片。默认保存全部帧并使用无损 PNG 格式，输出到输入文件旁的 `<文件名>_frames` 目录。

## 安装

```powershell
python -m pip install -r insta_pan_tools/requirements.txt
```

## 使用

```powershell
python insta_pan_tools/extract_frames.py "D:\video\VID_20260818_120000.insv"
```

指定输出目录并保存为 PNG：

```powershell
python insta_pan_tools/extract_frames.py `
    "D:\video\VID_20260818_120000.insv" `
    --output "D:\video\frames" `
    --format png
```

每 5 帧保存一帧：

```powershell
python insta_pan_tools/extract_frames.py "D:\video\VID_20260818_120000.insv" --every 5
```

如果需要压缩体积，可以改用有损 JPEG：

```powershell
python insta_pan_tools/extract_frames.py "D:\video\VID_20260818_120000.insv" `
    --format jpg `
    --jpeg-quality 95
```

默认输出文件名形如 `frame_000000.png`、`frame_000001.png`。不指定 `--overwrite` 时，输出目录中已存在的同名文件会被跳过。

也可以作为 Python 函数调用：

```python
from insta_pan_tools import extract_frames

result = extract_frames(
    r"D:\video\VID_20260818_120000.insv",
    image_format="png",
)
print(result.frames_saved, result.output_dir)
```

如果 `.insv` 无法打开，请确认当前安装的 OpenCV 包含 FFmpeg 解码支持；部分使用 HEVC 编码的文件还需要本机 FFmpeg/编解码器支持。
