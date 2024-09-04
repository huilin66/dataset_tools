import cv2
import time

# rgb,开机一段时间后问鼎在20fps
cap = cv2.VideoCapture('rtsp://admin:rxcj12345@192.168.1.139/video/pri')

# t, 开机一段时间后问鼎在30fps
# cap = cv2.VideoCapture('rtsp://192.168.1.100/video/pri')

for i in range(10):
    if cap.isOpened():
        # 手动计算 FPS
        num_frames = 120  # 计算 120 帧的实际 FPS
        start = time.time()

        for i in range(num_frames):
            ret, frame = cap.read()

        end = time.time()

        # 计算实际 FPS
        seconds = end - start
        fps = num_frames / seconds
        print(f"实际帧率为: {fps} FPS")
    else:
        print("无法打开视频流")

# 释放视频流
cap.release()