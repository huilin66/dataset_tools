import sys
import cv2
import threading
import time
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget


class VideoRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.recording = False
        self.rgb_cap = cv2.VideoCapture('rtsp://admin:rxcj12345@192.168.1.139/video/pri')
        self.thermal_cap = cv2.VideoCapture('rtsp://192.168.1.100/video/pri')
        self.rgb_out = None
        self.thermal_out = None
        self.recording_thread = None

    def initUI(self):
        self.setWindowTitle('Video Recorder')
        layout = QVBoxLayout()
        self.start_button = QPushButton('Start Recording')
        self.stop_button = QPushButton('Stop Recording')
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_recording(self):

        print('Start Recording')
        print(datetime.datetime.now())
        self.rgb_cap.release()
        self.thermal_cap.release()
        self.rgb_cap = cv2.VideoCapture('rtsp://admin:rxcj12345@192.168.1.139/video/pri')
        self.thermal_cap = cv2.VideoCapture('rtsp://192.168.1.100/video/pri')
        self.rgb_out = cv2.VideoWriter('rgb.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (2560, 1440))
        self.thermal_out = cv2.VideoWriter('t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
        self.recording = True
        self.recording_thread = threading.Thread(target=self.record_video_streams)
        self.recording_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        print('Stop Recording')
        print(datetime.datetime.now())
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def record_video_streams(self):
        start_time = time.time()
        while self.recording:
            ret_rgb, frame_rgb = self.rgb_cap.read()
            ret_thermal, frame_thermal = self.thermal_cap.read()
            if ret_rgb:
                self.rgb_out.write(frame_rgb)
            else:
                print("Error reading RGB frame.")
            if ret_thermal:
                self.thermal_out.write(frame_thermal)
            else:
                print("Error reading thermal frame.")

            # Log the current timestamp to diagnose timing issues
            current_time = time.time()
            print(f"Recording... Elapsed time: {current_time - start_time} seconds")

        end_time = time.time()
        print(f"Recording duration: {end_time - start_time} seconds")

        # Release resources when done
        self.rgb_cap.release()
        self.thermal_cap.release()
        self.rgb_out.release()
        self.thermal_out.release()
        cv2.destroyAllWindows()


def main():
    app = QApplication(sys.argv)
    recorder = VideoRecorder()
    recorder.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()