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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
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
        self.class_file = ""
        self.class_names = []  # 存储类别名称
        self.current_image_path = ""
        self.current_label_path = ""
        self.current_masks = []
        self.current_risks = []
        self.selected_mask_index = -1
        
        # 设置焦点策略，确保键盘事件能被捕获
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.init_ui()
        
        # 确保主窗口获得焦点
        self.setFocus()
        
        # 为所有子组件设置事件过滤器
        self.setEventFilter()
        
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
        self.image_display.mask_selected.connect(self.on_mask_selected)
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
        self.risk_editor.object_id_updated.connect(self.on_object_id_updated)
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
        
        # 加载类别文件
        load_class_action = file_menu.addAction('加载类别文件')
        load_class_action.triggered.connect(self.load_class_file)
        
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
        
        self.mask_list_widget = QListWidget()
        self.mask_list_widget.itemClicked.connect(self.on_mask_list_item_clicked)
        layout.addWidget(self.mask_list_widget)
        
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
            
    def load_class_file(self):
        """加载类别文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择类别文件", "", "文本文件 (*.txt)"
        )
        if file_path:
            self.class_file = file_path
            
            # 读取类别文件
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                
                self.statusBar().showMessage(f"已加载类别文件: {file_path}，共{len(self.class_names)}个类别")
                QMessageBox.information(self, "成功", f"已成功加载{len(self.class_names)}个类别")
            except Exception as e:
                self.statusBar().showMessage(f"加载类别文件失败: {str(e)}")
                QMessageBox.warning(self, "警告", f"加载类别文件失败: {str(e)}")
            
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
        
        # 提取图像名

        image_name = os.path.basename(self.current_image_path)
        
        # 显示图像和mask，并传递类别名称和图像名
        self.image_display.display_image_with_masks(image, masks, risks, self.class_names, image_name)
        
        # 更新mask信息
        self.update_mask_info()
        
        # 清除选中状态
        self.selected_mask_index = -1
        self.risk_editor.clear()
        
    def update_mask_info(self):
        """更新mask信息显示"""
        if not self.current_masks:
            self.mask_list_widget.clear()
            return
            
        self.mask_list_widget.clear()
        for i, (mask, risk) in enumerate(zip(self.current_masks, self.current_risks)):
            class_id = mask['class_id']
            class_name = f"Class {class_id}"
            object_id = mask['object_id']
            
            # 如果加载了类别文件，使用类别名称
            if self.class_names and class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            risk_levels = risk['risk_levels']
            
            item_text = f"id:{object_id} ; {class_name}"
            
            # 显示风险信息
            risk_names = []
            for j, level in enumerate(risk_levels):
                if level > 0:  # 只显示有风险的项目
                    level_name = {0: "No", 1: "Medium", 2: "High"}[level]
                    risk_types = ["abandonment", "broken", "corrosion", "deformation"]
                    risk_names.append(f"{risk_types[j]}: {level_name}")
                    
            if risk_names:
                item_text += f" ({', '.join(risk_names)})"
            else:
                item_text += " (无风险)"
            

            item = QListWidgetItem(item_text)
            self.mask_list_widget.addItem(item)
            
            # 如果是当前选中的mask，设置为选中状态
            if i == self.selected_mask_index:
                self.mask_list_widget.setCurrentItem(item)
        
    def on_mask_selected(self, mask_index):
        """mask选择事件"""
        if 0 <= mask_index < len(self.current_risks):
            self.selected_mask_index = mask_index
            risk = self.current_risks[mask_index]
            object_id = self.current_masks[mask_index]['object_id']
            self.risk_editor.set_risk_data(risk, object_id)
            # 更新mask列表选中状态
            self.update_mask_info()
        else:
            self.selected_mask_index = -1
            self.risk_editor.clear()
            # 更新mask列表选中状态
            self.update_mask_info()

    def on_mask_list_item_clicked(self, item):
        """mask列表项点击事件"""
        index = self.mask_list_widget.row(item)
        # 调用image_display的select_mask方法
        self.image_display.select_mask(index)
            
    def keyPressEvent(self, event):
        """键盘按下事件处理"""
        # 确保即使图像列表没有焦点，快捷键也能工作
        if event.key() == Qt.Key_A:
            self.show_previous_image()
            event.accept()
        elif event.key() == Qt.Key_D:
            self.show_next_image()
            event.accept()
        else:
            super().keyPressEvent(event)
            
    # 移除阻止焦点移动的代码，允许输入框获得焦点
    # 重写focusNextChild和focusPreviousChild方法，允许焦点在子组件间移动
    def focusNextChild(self):
        return super().focusNextChild()
        
    def focusPreviousChild(self):
        return super().focusPreviousChild()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        
    def setEventFilter(self):
        """为所有子组件设置事件过滤器，确保键盘事件能被主窗口捕获"""
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)
    
    def eventFilter(self, source, event):
        """事件过滤器，只处理快捷键而不干扰输入框"""
        if event.type() == QEvent.KeyPress:
            # 只处理特定的快捷键(A和D)，让其他按键事件正常传播
            if event.key() == Qt.Key_A or event.key() == Qt.Key_D:
                # 对于快捷键，调用主窗口的keyPressEvent
                self.keyPressEvent(event)
                return True
        return super().eventFilter(source, event)

    def event(self, event):
        return super().event(event)
            
    def show_previous_image(self):
        """显示上一个图像"""
        # 确保图像列表不为空
        if self.image_list.count() == 0:
            return
            
        current_row = self.image_list.currentRow()
        
        # 如果没有选中的项，默认选中最后一个
        if current_row == -1:
            last_row = self.image_list.count() - 1
            self.image_list.setCurrentRow(last_row)
            item = self.image_list.currentItem()
            if item:
                self.on_image_selected(item)
            return
            
        if current_row > 0:
            self.image_list.setCurrentRow(current_row - 1)
            item = self.image_list.currentItem()
            if item:
                self.on_image_selected(item)
        else:
            # 如果已经是第一个图像，则循环到最后一个
            last_row = self.image_list.count() - 1
            if last_row >= 0:
                self.image_list.setCurrentRow(last_row)
                item = self.image_list.currentItem()
                if item:
                    self.on_image_selected(item)
            
    def show_next_image(self):
        """显示下一个图像"""
        # 确保图像列表不为空
        if self.image_list.count() == 0:
            return
            
        current_row = self.image_list.currentRow()
        last_row = self.image_list.count() - 1
        
        # 如果没有选中的项，默认选中第一个
        if current_row == -1:
            self.image_list.setCurrentRow(0)
            item = self.image_list.currentItem()
            if item:
                self.on_image_selected(item)
            return
            
        if current_row < last_row:
            self.image_list.setCurrentRow(current_row + 1)
            item = self.image_list.currentItem()
            if item:
                self.on_image_selected(item)
        else:
            # 如果已经是最后一个图像，则循环到第一个
            if last_row >= 0:
                self.image_list.setCurrentRow(0)
                item = self.image_list.currentItem()
                if item:
                    self.on_image_selected(item)
            
    def on_risk_updated(self, risk_data):
        """风险信息更新事件"""
        if self.selected_mask_index >= 0 and self.selected_mask_index < len(self.current_risks):
            # 更新内存中的风险数据
            self.current_risks[self.selected_mask_index] = risk_data
            
            # 保存到文件
            self.save_updated_labels()
            
            # 更新显示
            self.update_mask_info()
            
            # 刷新图像显示
            image = cv2.imread(self.current_image_path)
            if image is not None:
                self.image_display.display_image_with_masks(image, self.current_masks, self.current_risks, self.class_names, self.image_display.current_image_name)

    def on_object_id_updated(self, new_id):
        """object_id更新事件"""
        if self.selected_mask_index >= 0 and self.selected_mask_index < len(self.current_masks):
            # 更新内存中的object_id
            old_id = self.current_masks[self.selected_mask_index]['object_id']
            self.current_masks[self.selected_mask_index]['object_id'] = new_id
            
            # 保存到文件
            self.save_updated_labels()
            
            # 更新显示
            self.update_mask_info()
            
            # 刷新图像显示
            image = cv2.imread(self.current_image_path)
            if image is not None:
                self.image_display.display_image_with_masks(image, self.current_masks, self.current_risks, self.class_names, self.image_display.current_image_name)
            
            self.statusBar().showMessage(f"Object ID已从 {old_id} 更新为 {new_id}")
            
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
    # pyinstaller --onefile --windowed main.py