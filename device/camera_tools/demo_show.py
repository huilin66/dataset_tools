import cv2

def video_show(camera_url):
    cap = cv2.VideoCapture(camera_url)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    print(frame_width, frame_height, frame_fps)
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


if __name__ == '__main__':
    pass
    # video_show("rtsp://192.168.1.100/video/pri")
    video_show("rtsp://admin:rxcj12345@192.168.1.139/video/pri")