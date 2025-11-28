from moviepy import VideoFileClip
import cv2
import os

def extract_frames(video_path, output_dir, interval=30):
    """
    video_path: 视频路径
    output_dir: 输出图片保存目录
    interval:   每隔多少帧保存一张图（例如 30 = 每秒约 1 帧，如果视频是 30fps）
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        if frame_idx % interval == 0:
            save_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Done! Saved {saved_idx} frames to {output_dir}")

def video_clip(input_path, output_path, start_time, end_time, left_crop, right_crop):
    # === 1. 读取并裁剪时间段 ===
    clip = VideoFileClip(input_path)

    # subclip 支持秒或 "mm:ss" 格式
    sub = clip.subclipped(start_time, end_time)

    # === 2. 根据宽度裁剪左右，只保留中间部分 ===
    w, h = sub.size  # 视频宽和高
    x1 = left_crop
    x2 = right_crop

    # 使用 crop 裁剪宽度，y 方向不裁剪（0 到 h）
    cropped = sub.cropped(x1=x1, x2=x2, y1=0, y2=h)

    # === 3. 导出视频 ===
    cropped.write_videofile(
        output_path,
        codec="libx264",     # 常用 H.264 编码
    )

    # 释放资源（可选）
    clip.close()
    sub.close()
    cropped.close()

if __name__ == '__main__':
    pass
    # input_path = r'/localnvme/data/bdd/DReality_data/video_data/V2_DJI_0731_W.MP4'
    # output_path = r'/localnvme/data/bdd/DReality_data/video_data/V2_DJI_0731_W_CLIP2.MP4'
    # output_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v1/images'
    # left_crop, right_crop = 700, 3000

    input_path = r'/localnvme/data/bdd/DReality_data/video_data/V6_DJI_0344_W.MP4'
    output_path = r'/localnvme/data/bdd/DReality_data/video_data/V6_DJI_0344_W_CLIP1.MP4'
    output_dir = r'/localnvme/data/bdd/DReality_data/yolo_clip_v2/images'
    start_time, end_time = 0, 60
    left_crop, right_crop = 500, 3000

    # video_clip(input_path, output_path, start_time, end_time, left_crop, right_crop)

    extract_frames(output_path, output_dir)