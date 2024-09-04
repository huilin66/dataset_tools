import sys
import cv2
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QStatusBar, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

class VideoStreamApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Stream Recorder")
        self.setGeometry(100, 100, 1200, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create top layout for video labels and streams
        self.top_layout = QVBoxLayout()
        self.layout.addLayout(self.top_layout)

        # Create top row layout for labels
        self.label_row = QHBoxLayout()
        self.top_layout.addLayout(self.label_row)

        # Create labels for video streams
        self.label_visible = QLabel("Visual Stream")
        self.label_ir = QLabel("Infrared Stream")
        self.label_visible.setFixedHeight(10)
        self.label_ir.setFixedHeight(10)
        self.label_row.addWidget(self.label_visible)
        self.label_row.addWidget(self.label_ir)

        # Create bottom row layout for video streams
        self.stream_row = QHBoxLayout()
        self.top_layout.addLayout(self.stream_row)

        # Create labels for video stream displays
        self.video_label_visible = QLabel()
        self.video_label_ir = QLabel()
        self.video_label_visible.setFixedHeight(500)
        self.video_label_ir.setFixedHeight(500)
        self.stream_row.addWidget(self.video_label_visible)
        self.stream_row.addWidget(self.video_label_ir)

        # Create path layout for visible video
        self.path_layout_visible = QHBoxLayout()
        self.layout.addLayout(self.path_layout_visible)

        # Create path input field and button for visible video
        self.path_input_visible = QLineEdit()
        self.path_input_visible.setPlaceholderText("Path to save visible video")
        self.path_button_visible = QPushButton("Browse...")
        self.path_button_visible.clicked.connect(self.browse_visible_path)
        self.path_layout_visible.addWidget(self.path_input_visible)
        self.path_layout_visible.addWidget(self.path_button_visible)

        # Create path layout for infrared video
        self.path_layout_ir = QHBoxLayout()
        self.layout.addLayout(self.path_layout_ir)

        # Create path input field and button for infrared video
        self.path_input_ir = QLineEdit()
        self.path_input_ir.setPlaceholderText("Path to save infrared video")
        self.path_button_ir = QPushButton("Browse...")
        self.path_button_ir.clicked.connect(self.browse_ir_path)
        self.path_layout_ir.addWidget(self.path_input_ir)
        self.path_layout_ir.addWidget(self.path_button_ir)

        # Create button layout for start and stop buttons
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # Create buttons
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize video capture and writer
        self.cap_visible = cv2.VideoCapture('rtsp://admin:rxcj12345@192.168.1.139/video/pri')
        self.cap_ir = cv2.VideoCapture('rtsp://192.168.1.100/video/pri')
        self.out_visible = None
        self.out_ir = None
        self.recording = False
        # self.shape_visible = [640, 360]
        self.shape_visible = [640, 480]
        self.shape_ir = [640, 480]

        # Create a timer to update the video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(0.1)  # 100 ms interval

    def browse_visible_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Visible Video", "", "Video Files (*.mp4)")
        if path:
            self.path_input_visible.setText(path)

    def browse_ir_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Infrared Video", "", "Video Files (*.mp4)")
        if path:
            self.path_input_ir.setText(path)

    def start_recording(self):
        self.recording = True
        self.status_bar.showMessage("Recording")

        visible_path = self.path_input_visible.text()
        ir_path = self.path_input_ir.text()

        if not visible_path or not ir_path:
            print("Please provide paths for both videos.")
            return

        if not self.cap_visible.isOpened() or not self.cap_ir.isOpened():
            print("Error opening video streams.")
            self.status_bar.showMessage("Error opening video streams.")
            return

        # Get frame properties
        width_visible = int(self.cap_visible.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_visible = int(self.cap_visible.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_ir = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_ir = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.out_visible = cv2.VideoWriter(visible_path, fourcc, 20.0, (width_visible, height_visible))
        self.out_ir = cv2.VideoWriter(ir_path, fourcc, 20.0, (width_ir, height_ir))

        # Store the aspect ratios
        self.aspect_ratio_visible = width_visible / height_visible
        self.aspect_ratio_ir = width_ir / height_ir

    def stop_recording(self):
        self.recording = False
        self.status_bar.showMessage("Ready")

        if self.out_visible:
            self.out_visible.release()
        if self.out_ir:
            self.out_ir.release()

    def update_frames(self):
        if self.cap_visible and self.cap_ir:

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            ret_visible, frame_visible = self.cap_visible.read()
            fps_visible = "%.1f"%self.cap_visible.get(cv2.CAP_PROP_FPS)
            ret_ir, frame_ir = self.cap_ir.read()
            fps_ir = "%.1f"%self.cap_ir.get(cv2.CAP_PROP_FPS)


            if not ret_visible:
                print("Error: Failed to capture visible frame.")
            else:
                if self.recording:
                    self.out_visible.write(frame_visible)
                self.display_image(self.video_label_visible, frame_visible, self.shape_visible, fps_visible, current_time)

            if not ret_ir:
                print("Error: Failed to capture infrared frame.")
            else:
                if self.recording:
                    self.out_ir.write(frame_ir)
                self.display_image(self.video_label_ir, frame_ir, self.shape_ir, fps_ir, current_time)

    def display_image(self, label, img, target_shape, fps_text, time_text):
        """Convert OpenCV image to QImage, resize it, and display it on QLabel."""
        if img is None:
            return

        # Resize image to target dimensions
        img = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)

        # 在图像上叠加显示 FPS 和当前时间
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoStreamApp()
    window.show()
    sys.exit(app.exec_())
