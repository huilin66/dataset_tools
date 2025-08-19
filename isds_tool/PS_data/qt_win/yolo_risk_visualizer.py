import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton, QGroupBox, QGridLayout, QFileDialog, QMenu, QMenuBar, QAction, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

class YOLORiskVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO Instance Segmentation Risk Visualizer')
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.image_folder = ''
        self.label_folder = ''
        self.image_files = []
        self.current_image_index = -1
        self.masks = []
        self.categories = []
        self.risk_values = []
        self.selected_mask_index = -1

        # Create UI
        self.init_ui()

    def init_ui(self):
        # Create menu bar
        self.create_menu_bar()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create top splitter (image display and image list)
        top_splitter = QSplitter(Qt.Horizontal)

        # Image display area
        self.image_label = QLabel('No image loaded')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        top_splitter.addWidget(self.image_label)

        # Image list area
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_selected)
        self.image_list_widget.setMinimumWidth(200)
        top_splitter.addWidget(self.image_list_widget)

        # Set initial sizes for top splitter
        top_splitter.setSizes([800, 200])

        # Create bottom splitter (risk details and mask list)
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Risk details area
        self.risk_group = QGroupBox('Risk Details')
        self.init_risk_group()
        bottom_splitter.addWidget(self.risk_group)

        # Mask list area
        self.mask_list_widget = QListWidget()
        self.mask_list_widget.itemClicked.connect(self.on_mask_selected)
        self.mask_list_widget.setMinimumWidth(200)
        bottom_splitter.addWidget(self.mask_list_widget)

        # Set initial sizes for bottom splitter
        bottom_splitter.setSizes([400, 400])

        # Add splitters to main layout
        main_layout.addWidget(top_splitter, 7)
        main_layout.addWidget(bottom_splitter, 3)

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        # Load image folder action
        load_images_action = QAction('Load Image Folder', self)
        load_images_action.triggered.connect(self.load_image_folder)
        file_menu.addAction(load_images_action)

        # Load label folder action
        load_labels_action = QAction('Load Label Folder', self)
        load_labels_action.triggered.connect(self.load_label_folder)
        file_menu.addAction(load_labels_action)

        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def init_risk_group(self):
        layout = QGridLayout()

        # Risk 1
        layout.addWidget(QLabel('Risk 1:'), 0, 0)
        self.risk1_no = QRadioButton('No')
        self.risk1_medium = QRadioButton('Medium')
        self.risk1_high = QRadioButton('High')
        layout.addWidget(self.risk1_no, 0, 1)
        layout.addWidget(self.risk1_medium, 0, 2)
        layout.addWidget(self.risk1_high, 0, 3)

        # Risk 2
        layout.addWidget(QLabel('Risk 2:'), 1, 0)
        self.risk2_no = QRadioButton('No')
        self.risk2_medium = QRadioButton('Medium')
        self.risk2_high = QRadioButton('High')
        layout.addWidget(self.risk2_no, 1, 1)
        layout.addWidget(self.risk2_medium, 1, 2)
        layout.addWidget(self.risk2_high, 1, 3)

        # Risk 3
        layout.addWidget(QLabel('Risk 3:'), 2, 0)
        self.risk3_no = QRadioButton('No')
        self.risk3_medium = QRadioButton('Medium')
        self.risk3_high = QRadioButton('High')
        layout.addWidget(self.risk3_no, 2, 1)
        layout.addWidget(self.risk3_medium, 2, 2)
        layout.addWidget(self.risk3_high, 2, 3)

        # Risk 4
        layout.addWidget(QLabel('Risk 4:'), 3, 0)
        self.risk4_no = QRadioButton('No')
        self.risk4_medium = QRadioButton('Medium')
        self.risk4_high = QRadioButton('High')
        layout.addWidget(self.risk4_no, 3, 1)
        layout.addWidget(self.risk4_medium, 3, 2)
        layout.addWidget(self.risk4_high, 3, 3)

        # Update button
        self.update_button = QPushButton('Update')
        self.update_button.clicked.connect(self.update_risk_values)
        layout.addWidget(self.update_button, 4, 0, 1, 4)

        self.risk_group.setLayout(layout)

    def load_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Image Folder')
        if folder:
            self.image_folder = folder
            self.load_image_list()
            self.statusBar().showMessage(f'Loaded images from: {folder}')

    def load_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Label Folder')
        if folder:
            self.label_folder = folder
            self.statusBar().showMessage(f'Loaded labels from: {folder}')

    def load_image_list(self):
        self.image_files = []
        self.image_list_widget.clear()
        if not self.image_folder:
            return

        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in extensions):
                self.image_files.append(file)
                self.image_list_widget.addItem(file)

    def on_image_selected(self, item):
        self.current_image_index = self.image_list_widget.row(item)
        self.load_image_and_masks()

    def load_image_and_masks(self):
        if not self.image_folder or not self.label_folder or self.current_image_index < 0:
            return

        # Get current image path
        image_name = self.image_files[self.current_image_index]
        image_path = os.path.join(self.image_folder, image_name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.image_label.setText('Failed to load image')
            return

        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Load corresponding label file
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_name)

        self.masks = []
        self.categories = []
        self.risk_values = []
        self.mask_list_widget.clear()

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Parse the modified YOLO format with risk attributes
                parts = list(map(float, line.split()))
                if len(parts) < 6:
                    continue  # Invalid format

                cat_id = int(parts[0])
                risk_num = int(parts[1])
                risks = parts[2:2+risk_num]
                polygon = np.array(parts[2+risk_num:]).reshape(-1, 2)

                # Store mask data
                self.masks.append(polygon.astype(int))
                self.categories.append(cat_id)
                self.risk_values.append(risks)

                # Add to mask list with risk info
                risk_text = ', '.join([f'Risk {i+1}: {self.get_risk_text(int(r))}' for i, r in enumerate(risks) if int(r) > 0])
                item_text = f'Category {cat_id}'
                if risk_text:
                    item_text += f' ({risk_text})'
                self.mask_list_widget.addItem(item_text)

        # Draw masks on image
        self.draw_masks(q_image, image, image_rgb.shape[:2])

    def draw_masks(self, q_image, original_image, image_shape):
        # Create a painter to draw masks
        painter = QPainter(q_image)
        painter.setRenderHint(QPainter.Antialiasing)

        # Define colors for different categories (simplified)
        colors = [
            QColor(255, 0, 0, 128),    # Red
            QColor(0, 255, 0, 128),    # Green
            QColor(0, 0, 255, 128),    # Blue
            QColor(255, 255, 0, 128),  # Yellow
            QColor(255, 0, 255, 128),  # Magenta
            QColor(0, 255, 255, 128),  # Cyan
        ]

        # Draw each mask
        for i, polygon in enumerate(self.masks):
            color = colors[i % len(colors)]
            painter.setBrush(color)
            painter.setPen(QPen(color, 2))

            # Create QPolygon from numpy array
            q_polygon = [QPoint(int(p[0]), int(p[1])) for p in polygon]
            painter.drawPolygon(q_polygon)

            # Draw category label
            label_pos = QPoint(int(polygon[0][0]), int(polygon[0][1]) - 10)
            painter.setPen(QPen(Qt.Black, 2))
            painter.drawText(label_pos, f'Cat {self.categories[i]}')

        painter.end()

        # Display the image with masks
        self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def on_mask_selected(self, item):
        self.selected_mask_index = self.mask_list_widget.row(item)
        if self.selected_mask_index < 0 or self.selected_mask_index >= len(self.risk_values):
            return

        # Get risk values for selected mask
        risks = self.risk_values[self.selected_mask_index]

        # Update radio buttons
        self.clear_risk_radios()
        for i, risk in enumerate(risks[:4]):  # Show up to 4 risks
            risk_val = int(risk)
            if i == 0:
                if risk_val == 0: self.risk1_no.setChecked(True)
                elif risk_val == 1: self.risk1_medium.setChecked(True)
                elif risk_val == 2: self.risk1_high.setChecked(True)
            elif i == 1:
                if risk_val == 0: self.risk2_no.setChecked(True)
                elif risk_val == 1: self.risk2_medium.setChecked(True)
                elif risk_val == 2: self.risk2_high.setChecked(True)
            elif i == 2:
                if risk_val == 0: self.risk3_no.setChecked(True)
                elif risk_val == 1: self.risk3_medium.setChecked(True)
                elif risk_val == 2: self.risk3_high.setChecked(True)
            elif i == 3:
                if risk_val == 0: self.risk4_no.setChecked(True)
                elif risk_val == 1: self.risk4_medium.setChecked(True)
                elif risk_val == 2: self.risk4_high.setChecked(True)

    def clear_risk_radios(self):
        # Clear all radio button selections
        self.risk1_no.setChecked(False)
        self.risk1_medium.setChecked(False)
        self.risk1_high.setChecked(False)
        self.risk2_no.setChecked(False)
        self.risk2_medium.setChecked(False)
        self.risk2_high.setChecked(False)
        self.risk3_no.setChecked(False)
        self.risk3_medium.setChecked(False)
        self.risk3_high.setChecked(False)
        self.risk4_no.setChecked(False)
        self.risk4_medium.setChecked(False)
        self.risk4_high.setChecked(False)

    def update_risk_values(self):
        if self.selected_mask_index < 0 or self.selected_mask_index >= len(self.risk_values):
            return

        # Get new risk values from radio buttons
        new_risks = []
        if self.risk1_no.isChecked(): new_risks.append(0)
        elif self.risk1_medium.isChecked(): new_risks.append(1)
        elif self.risk1_high.isChecked(): new_risks.append(2)

        if self.risk2_no.isChecked(): new_risks.append(0)
        elif self.risk2_medium.isChecked(): new_risks.append(1)
        elif self.risk2_high.isChecked(): new_risks.append(2)

        if self.risk3_no.isChecked(): new_risks.append(0)
        elif self.risk3_medium.isChecked(): new_risks.append(1)
        elif self.risk3_high.isChecked(): new_risks.append(2)

        if self.risk4_no.isChecked(): new_risks.append(0)
        elif self.risk4_medium.isChecked(): new_risks.append(1)
        elif self.risk4_high.isChecked(): new_risks.append(2)

        # Update risk values
        self.risk_values[self.selected_mask_index] = new_risks

        # Save to label file
        self.save_label_file()

        # Refresh mask list
        self.refresh_mask_list()

    def save_label_file(self):
        if not self.image_folder or not self.label_folder or self.current_image_index < 0:
            return

        # Get current image name
        image_name = self.image_files[self.current_image_index]
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_name)

        # Write updated labels
        with open(label_path, 'w') as f:
            for i in range(len(self.categories)):
                cat_id = self.categories[i]
                risk_num = len(self.risk_values[i])
                risks = ' '.join(map(str, self.risk_values[i]))
                polygon = ' '.join(map(str, self.masks[i].flatten()))
                line = f'{cat_id} {risk_num} {risks} {polygon}\n'
                f.write(line)

    def refresh_mask_list(self):
        self.mask_list_widget.clear()
        for i in range(len(self.categories)):
            # Add to mask list with risk info
            risk_text = ', '.join([f'Risk {j+1}: {self.get_risk_text(int(r))}' for j, r in enumerate(self.risk_values[i]) if int(r) > 0])
            item_text = f'Category {self.categories[i]}'
            if risk_text:
                item_text += f' ({risk_text})'
            self.mask_list_widget.addItem(item_text)

    def get_risk_text(self, risk_value):
        if risk_value == 0: return 'No'
        elif risk_value == 1: return 'Medium'
        elif risk_value == 2: return 'High'
        return 'Unknown'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLORiskVisualizer()
    window.show()
    sys.exit(app.exec_())