#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Instance Segmentation Risk Annotator
支持risk标签的YOLO实例分割标注工具
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QListWidget, QListWidgetItem, QFileDialog, 
                             QMenuBar, QStatusBar, QMessageBox, QComboBox,
                             QGroupBox, QCheckBox, QSpinBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont, QImage

from yolo_parser import YOLOParser
from image_display import ImageDisplayWidget
from risk_editor import RiskEditorWidget


class YOLORiskAnnotator(QMainWindow):
    """YOLO实例分割风险标注工具主窗口"""
    
    def __init__(self):
        super().__init__()
        self.image_folder = ""
        self.label_folder = ""
        self.current_image_path = ""
        self.current_label_path = ""
        self.current_masks = []
        self.current_risks = []
        self.selected_mask_index = -1
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("YOLO Instance Segmentation Risk Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar()
        
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QGridLayout(central_widget)
        
        # 左上角：图像显示区域
        self.image_display = ImageDisplayWidget()
        main_layout.addWidget(self.image_display, 0, 0, 2, 1)
        
        # 右上角：图像文件列表
        self.create_image_list_widget()
        main_layout.addWidget(self.image_list_group, 0, 1, 1, 1)
        
        # 右下角：当前图像mask信息
        self.create_mask_info_widget()
        main_layout.addWidget(self.mask_info_group, 1, 1, 1, 1)
        
        # 左下角：风险编辑区域
        self.risk_editor = RiskEditorWidget()
        self.risk_editor.risk_updated.connect(self.on_risk_updated)
        main_layout.addWidget(self.risk_editor, 2, 0, 1, 1)
        
        # 设置布局比例
        main_layout.setColumnStretch(0, 3)  # 图像显示区域占3份
        main_layout.setColumnStretch(1, 1)  # 右侧区域占1份
        main_layout.setRowStretch(0, 2)     # 上部分占2份
        main_layout.setRowStretch(1, 1)     # 下部分占1份
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        # 加载图像文件夹
        load_images_action = file_menu.addAction('加载图像文件夹')
        load_images_action.triggered.connect(self.load_image_folder)
        
        # 加载标签文件夹
        load_labels_action = file_menu.addAction('加载标签文件夹')
        load_labels_action.triggered.connect(self.load_label_folder)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)
        
    def create_image_list_widget(self):
        """创建图像列表部件"""
        self.image_list_group = QGroupBox("图像文件列表")
        layout = QVBoxLayout(self.image_list_group)
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
    def create_mask_info_widget(self):
        """创建mask信息显示部件"""
        self.mask_info_group = QGroupBox("当前图像Mask信息")
        layout = QVBoxLayout(self.mask_info_group)
        
        self.mask_info_text = QTextEdit()
        self.mask_info_text.setReadOnly(True)
        layout.addWidget(self.mask_info_text)
        
    def load_image_folder(self):
        """加载图像文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.image_folder = folder
            self.load_image_list()
            self.statusBar().showMessage(f"已加载图像文件夹: {folder}")
            
    def load_label_folder(self):
        """加载标签文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择标签文件夹")
        if folder:
            self.label_folder = folder
            self.statusBar().showMessage(f"已加载标签文件夹: {folder}")
            
    def load_image_list(self):
        """加载图像文件列表"""
        if not self.image_folder:
            return
            
        self.image_list.clear()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                item = QListWidgetItem(file)
                self.image_list.addItem(item)
                
    def on_image_selected(self, item):
        """图像选择事件"""
        image_name = item.text()
        self.current_image_path = os.path.join(self.image_folder, image_name)
        
        # 查找对应的标签文件
        label_name = os.path.splitext(image_name)[0] + '.txt'
        self.current_label_path = os.path.join(self.label_folder, label_name)
        
        # 加载并显示图像和标签
        self.load_image_and_labels()
        
    def load_image_and_labels(self):
        """加载图像和标签"""
        if not os.path.exists(self.current_image_path):
            return
            
        # 加载图像
        image = cv2.imread(self.current_image_path)
        if image is None:
            return
            
        # 解析标签文件
        parser = YOLOParser()
        masks, risks = parser.parse_label_file(self.current_label_path, image.shape)
        
        self.current_masks = masks
        self.current_risks = risks
        
        # 显示图像和mask
        self.image_display.display_image_with_masks(image, masks, risks)
        
        # 更新mask信息
        self.update_mask_info()
        
        # 清除选中状态
        self.selected_mask_index = -1
        self.risk_editor.clear()
        
    def update_mask_info(self):
        """更新mask信息显示"""
        if not self.current_masks:
            self.mask_info_text.clear()
            return
            
        info_text = ""
        for i, (mask, risk) in enumerate(zip(self.current_masks, self.current_risks)):
            class_id = mask['class_id']
            risk_levels = risk['risk_levels']
            
            info_text += f"Mask {i+1}:\n"
            info_text += f"  类别ID: {class_id}\n"
            
            # 显示风险信息
            risk_names = []
            for j, level in enumerate(risk_levels):
                if level > 0:  # 只显示有风险的项目
                    level_name = {0: "No", 1: "Medium", 2: "High"}[level]
                    risk_names.append(f"Risk{j+1}: {level_name}")
                    
            if risk_names:
                info_text += f"  风险: {', '.join(risk_names)}\n"
            else:
                info_text += f"  风险: 无\n"
            info_text += "\n"
            
        self.mask_info_text.setText(info_text)
        
    def on_mask_selected(self, mask_index):
        """mask选择事件"""
        if 0 <= mask_index < len(self.current_risks):
            self.selected_mask_index = mask_index
            risk = self.current_risks[mask_index]
            self.risk_editor.set_risk_data(risk)
            
    def on_risk_updated(self, risk_data):
        """风险信息更新事件"""
        if self.selected_mask_index >= 0 and self.selected_mask_index < len(self.current_risks):
            # 更新内存中的风险数据
            self.current_risks[self.selected_mask_index] = risk_data
            
            # 保存到文件
            self.save_updated_labels()
            
            # 更新显示
            self.update_mask_info()
            
    def save_updated_labels(self):
        """保存更新后的标签文件"""
        if not self.current_label_path or not self.current_masks:
            return
            
        parser = YOLOParser()
        parser.save_label_file(self.current_label_path, self.current_masks, self.current_risks)
        self.statusBar().showMessage("标签文件已更新")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = YOLORiskAnnotator()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()