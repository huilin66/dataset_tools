import cv2
import threading

rtsp_url_ir = 'rtsp://192.168.1.100/video/pri'  # 热红外视频流
rtsp_url_visible = 'rtsp://admin:rxcj12345@192.168.1.139/video/pri'  # 可见光视频流

# 定义摄像头的 URL
# camera_url = "http://admin:rxcj12345@192.168.1.100:10080"
# camera_url = "http://admin:rxcj12345@192.168.1.100"
# camera_url = "http://192.168.1.100:10080"
# camera_url = "http://192.168.1.100"

# camera_ip = "192.168.1.100:10080"
# username = 'admin'
# password = 'rxcj12345'

# camera_url = f"rtsp://{username}:{password}@{camera_ip}/video/pri"
# camera_url = f"http://{username}:{password}@{camera_ip}/video/pri"
# camera_url = f"rtsp://{camera_ip}/video/pri"
# camera_url = f"http://{camera_ip}/video/pri"
# camera_url = "http://192.168.1.100:10090/media/addStreamProxy?secret=k46RfnEGwg4UMWckvppKNEjvaM38euBm&vhost=192.168.1.100&app=video&stream=pri&url=rtsp://192.168.1.100/video/pri"
# camera_url = "rtsp://192.168.1.100/video/pri"
# camera_url = "rtsp://192.168.1.100/video/rgb"
'''
rtsp://192.168.1.100/video/pri

http://192.168.1.100:10090/media/addStreamProxy?secret=k46RfnEGwg4UMWckvppKNEjvaM38euBm&vhost=192.168.1.100&app=video&stream=pri&url=rtsp://192.168.1.100/video/pri
'''

def video_show(camera_url):
    cap = cv2.VideoCapture(camera_url)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_width, frame_height)
    if not cap.isOpened():
        print("无法打开摄像头")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Infrared Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

def video_save(camera_url, video_path=None, img_dir=None):
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print("无法打开摄像头")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Infrared Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()







# 视频录制线程函数
def record_video(rtsp_url, output_file, codec):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"无法打开视频流: {rtsp_url}")
        return

    # 获取视频流的帧宽和帧高
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_file, codec, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"视频流已断开: {rtsp_url}")
            break

        # 写入视频帧
        out.write(frame)

        # 显示视频流（可选）
        cv2.imshow(f'Recording: {output_file}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def record_videos(rtsp_url_ir, rtsp_url_visible, output_file_ir, output_file_visible):


    # 创建并启动线程
    thread_visible = threading.Thread(target=record_video, args=(rtsp_url_visible, output_file_visible, cv2.VideoWriter_fourcc(*'mp4v')))
    thread_ir = threading.Thread(target=record_video, args=(rtsp_url_ir, output_file_ir, cv2.VideoWriter_fourcc(*'mp4v')))
    # thread_visible = threading.Thread(target=record_video, args=(rtsp_url_visible, output_file_visible, cv2.VideoWriter_fourcc(*'mp4v')))

    thread_visible.start()
    thread_ir.start()
    # thread_visible.start()

    # 等待线程完成
    thread_visible.join()
    thread_ir.join()
    # thread_visible.join()


if __name__ == '__main__':
    pass
    # video_show("rtsp://192.168.1.100:10080/video/b7764dd875ec4a1aa80a88ed0e1a1969")
    # video_show("rtsp://192.168.1.139:8000/video")
    video_show("rtsp://192.168.1.100/video/pri")
    # video_show("rtsp://admin:rxcj12345@192.168.1.139/video/pri")


    # 视频流 URL
    # rtsp_url_ir = 'rtsp://192.168.1.100/video/pri'  # 热红外视频流
    # rtsp_url_visible = 'rtsp://admin:rxcj12345@192.168.1.139/video/pri'  # 可见光视频流



    # # 视频流 URL
    # rtsp_url_ir = 'rtsp://192.168.1.100/video/pri'  # 热红外视频流
    # rtsp_url_visible = 'rtsp://admin:rxcj12345@192.168.1.139/video/pri'  # 可见光视频流
    #
    # # 保存视频的文件名
    # output_file_ir = 'thermal_video.mp4'
    # output_file_visible = 'visible_video.mp4'
    #
    # record_videos(rtsp_url_ir, rtsp_url_visible, output_file_ir, output_file_visible)

