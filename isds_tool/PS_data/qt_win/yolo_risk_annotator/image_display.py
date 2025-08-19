#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像显示组件
支持显示图像和实例分割mask，并支持点击选择mask
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, 
                             QHBoxLayout, QCheckBox, QListWidgetItem, QPushButton,
                             QGridLayout)


from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage, QMouseEvent

class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox, self).__init__()
        self.img = None
        self.scaled_img = None
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.wheel_flag = False
 
        self.scale = 1
        self.old_scale = 1
        self.point = QPoint(0, 0)
        self.x = -1
        self.y = -1
        self.new_height = -1
        self.new_width = -1
 
    def init_ui(self):
        self.setWindowTitle("ImageBox")
 
    def set_image(self, img_path):
        self.img = QPixmap(img_path)
        width, height = self.img.width(), self.img.height()
        if height / width > 990 / 660:
            new_height = 990
            new_width = width * 990 / height
        else:
            new_height = height * 660 / width
            new_width = 660
        self.point = QPoint(int((660 - new_width) * 0.5), int((990 - new_height) * 0.5))
        self.img = self.img.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.scaled_img = self.img
 
        self.new_height = new_height
        self.new_width = new_width
        self.scale = 1
 
    def paintEvent(self, e):
        if self.scaled_img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale, self.scale)
            if self.wheel_flag:        # 定点缩放
                self.wheel_flag = False
                # 判断当前鼠标pos在不在图上
                this_left_x = self.point.x() * self.old_scale
                this_left_y = self.point.y() * self.old_scale
                this_scale_width = self.new_width * self.old_scale
                this_scale_height = self.new_height * self.old_scale
 
                # 鼠标点在图上，以鼠标点为中心动作
                gap_x = self.x - this_left_x
                gap_y = self.y - this_left_y
                if 0 < gap_x < this_scale_width and 0 < gap_y < this_scale_height:
                    new_left_x = int(self.x / self.scale - gap_x / self.old_scale)
                    new_left_y = int(self.y / self.scale - gap_y / self.old_scale)
                    self.point = QPoint(new_left_x, new_left_y)
                # 鼠标点不在图上，固定左上角进行缩放
                else:
                    true_left_x = int(self.point.x() * self.old_scale / self.scale)
                    true_left_y = int(self.point.y() * self.old_scale / self.scale)
                    self.point = QPoint(true_left_x, true_left_y)
            painter.drawPixmap(self.point, self.scaled_img)  # 此函数中还会用scale对point进行处理
            painter.end()
 
    def wheelEvent(self, event):
        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleY = angle.y()
        self.old_scale = self.scale
        self.x, self.y = event.x(), event.y()
        self.wheel_flag = True
        # 获取当前鼠标相对于view的位置
        if angleY > 0:
            self.scale *= 1.08
        else:  # 滚轮下滚
            self.scale *= 0.92
        if self.scale < 0.3:
            self.scale = 0.3
        self.adjustSize()
        self.update()
 
    def mouseMoveEvent(self, e):
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos                    # 当前位置-起始位置=差值
            self.point = self.point + self.end_pos / self.scale        # 左上角的距离变化
            self.start_pos = e.pos()
            self.repaint()
 
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()
 
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.left_click = False
 
class ImageDisplayWidget(QWidget):
    """图像显示组件"""
    
    mask_selected = pyqtSignal(int)  # 发送选中的mask索引
    
    def __init__(self):
        super().__init__()
        self.image_box = ImageBox()  # 初始化ImageBox实例
        self.current_image = None
        self.original_image = None
        self.current_masks = []
        self.current_risks = []
        self.selected_mask_index = -1
        self.class_names = []  # 存储类别名称
        self.show_masks = True  # 是否显示mask和标签
        self.current_image_name = ""  # 当前图像文件名
        
        self.init_ui()
        
        # 添加更新风险信息的快捷键
        from PyQt5.QtWidgets import QAction
        update_risk_action = QAction('更新风险信息', self)
        update_risk_action.setShortcut('U')
        update_risk_action.triggered.connect(self.update_risk_info)
        self.addAction(update_risk_action)
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建控制区域（放在最上方）
        control_layout = QHBoxLayout()
        self.show_masks_checkbox = QCheckBox("显示标签和mask")
        self.show_masks_checkbox.setChecked(True)
        self.show_masks_checkbox.stateChanged.connect(self.on_show_masks_changed)
        control_layout.addWidget(self.show_masks_checkbox)
        
        # 添加缩放到全局按钮
        self.zoom_to_global_btn = QPushButton("缩放到全局")
        self.zoom_to_global_btn.clicked.connect(self.zoom_to_global)
        control_layout.addWidget(self.zoom_to_global_btn)
        
        # 添加图像名显示和复制按钮
        self.image_name_label = QLabel("无图像")
        self.image_name_label.setStyleSheet("margin-left: 10px; color: blue;")
        control_layout.addWidget(self.image_name_label)
        
        self.copy_name_btn = QPushButton("复制名")
        self.copy_name_btn.clicked.connect(self.copy_image_name)
        control_layout.addWidget(self.copy_name_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # 添加ImageBox显示区域（在控制区域下方）
        self.image_box.setMinimumSize(400, 300)
        layout.addWidget(self.image_box)
        
        # 创建风险信息标签
        self.risk_info_label = QLabel("未选择对象")
        self.risk_info_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        self.risk_info_label.setStyleSheet("background-color: rgba(0, 0, 0, 0.5); color: white; padding: 5px; max-height: 40px;")
        self.risk_info_label.setWordWrap(True)
        layout.addWidget(self.risk_info_label)
        
        # 连接信号
        self.mask_selected.connect(self.on_mask_selected)
        
        # 不覆盖ImageBox的鼠标事件处理，而是使用事件传播
        # 这样可以保持ImageBox原有的拖动功能
        pass
        
    def on_show_masks_changed(self, state):
        """显示mask和标签复选框状态改变事件"""
        self.show_masks = state == Qt.Checked
        self.redraw_image()
        
    def display_image_with_masks(self, image: np.ndarray, masks: List[Dict], risks: List[Dict], class_names: Optional[List[str]] = None, image_name: str = ""):
        """
        显示图像和mask
        
        Args:
            image: 图像数组
            masks: mask信息列表
            risks: risk信息列表
            class_names: 类别名称列表，可选
            image_name: 图像文件名，可选
        """
        # 保存原始数据
        self.original_image = image.copy()
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        self.current_masks = masks
        self.current_risks = risks
        self.selected_mask_index = -1
        self.current_image_name = image_name
        
        # 更新图像名显示
        self.image_name_label.setText(self.current_image_name if self.current_image_name else "无图像")
        
        if class_names is not None:
            self.class_names = class_names
        
        # 创建带标签的显示图像
        display_image = self.create_display_image()
        
        # 转换为QPixmap
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 设置图像到ImageBox
        self.image_box.img = pixmap
        self.image_box.scaled_img = pixmap
        self.image_box.new_width = width
        self.image_box.new_height = height
        
        # 缩放到全局
        self.zoom_to_global()
        
    def create_display_image(self) -> np.ndarray:
        if self.original_image is None:
            return np.zeros((300, 400, 3), dtype=np.uint8)
            
        # 复制原图像
        display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).copy()
        
        # 定义颜色映射
        colormap = [
            (255, 42, 4), (183, 223, 0), (104, 31, 17), (221, 111, 255),
            (79, 68, 255), (0, 237, 204), (68, 243, 0), (255, 0, 189)
        ]
        colors = colormap
        
        # 根据show_masks标志决定是否绘制mask和标签
        if self.show_masks:
            # 绘制每个mask
            for i, (mask, risk) in enumerate(zip(self.current_masks, self.current_risks)):
                coordinates = mask['coordinates']
                if len(coordinates) < 3:
                    continue
                    
                # 选择颜色
                color = colors[i % len(colors)]
                
                # 将归一化坐标转换为像素坐标
                height, width = display_image.shape[:2]
                pixel_coordinates = [(int(x * width), int(y * height)) for x, y in coordinates]
                points = np.array(pixel_coordinates, dtype=np.int32)
                
                # 绘制mask边界
                cv2.polylines(display_image, [points], True, color, 2)
                
                # 如果是选中的mask，绘制特殊标记
                if i == self.selected_mask_index:
                    cv2.polylines(display_image, [points], True, (255, 255, 255), 3)
                    
                # 添加标签
                class_id = mask['class_id']
                risk_desc = self.get_risk_description(risk['risk_levels'])
                
                # 使用类别名称
                if self.class_names and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                    
                label = class_name
                if risk_desc != "无风险":
                    label += f" ({risk_desc})"
                    
                # 计算标签位置
                x, y = coordinates[0]
                x_pixel = int(x * width)
                y_pixel = int(y * height)
                
                # 绘制标签
                cv2.putText(display_image, label, (x_pixel, y_pixel - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return display_image
        
    def get_risk_description(self, risk_levels: List[int]) -> str:
        """获取risk描述"""
        risk_names = ["abandonment", "broken", "corrosion", "deformation"]
        risk_level_names = {0: "No", 1: "Medium", 2: "High"}
        
        descriptions = []
        for i, level in enumerate(risk_levels):
            if level > 0:
                level_name = risk_level_names[level]
                descriptions.append(f"{risk_names[i]}: {level_name}")
                
        return ", ".join(descriptions) if descriptions else "无风险"
        


    def wheelEvent(self, event):
        self.image_box.wheelEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件处理"""
        # 先调用父类方法让事件自然传播
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件处理"""
        # 先调用父类方法让事件自然传播
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        mask_clicked = False
        
        if self.current_masks and event.button() == Qt.LeftButton:
            # 获取点击位置
            pos = event.pos()
            
            # 找到点击的mask
            clicked_mask_index = self.find_clicked_mask(pos)
            
            if clicked_mask_index != -1:
                self.selected_mask_index = clicked_mask_index
                self.mask_selected.emit(clicked_mask_index)
                
                # 重新绘制图像
                self.redraw_image()
                mask_clicked = True
            else:
                # 取消选择
                self.selected_mask_index = -1
                self.mask_selected.emit(-1)
                self.redraw_image()
        
        # 如果没有点击到mask，让事件自然传播
        if not mask_clicked:
            return super().mousePressEvent(event)
        
    # 移除旧方法，不再需要
    # def on_mouse_press(self, event: QMouseEvent):
    #     self.mousePressEvent(event)
            
    def find_clicked_mask(self, pos: QPoint) -> int:
        """找到点击的mask"""
        if not self.current_masks or self.current_image is None:
            return -1
            
        # 转换坐标到图像坐标系
        # 使用image_box的属性进行计算
        box_size = self.image_box.size()
        
        # 获取图像的原始尺寸
        height, width = self.current_image.shape[:2]
        
        # 计算缩放比例
        scale = self.image_box.scale
        
        # 转换点击坐标（考虑图像在box中的偏移和缩放）
        x = int((pos.x() - self.image_box.point.x() * scale) / scale)
        y = int((pos.y() - self.image_box.point.y() * scale) / scale)
        
        # 检查每个mask
        for i, mask in enumerate(self.current_masks):
            coordinates = mask['coordinates']
            if len(coordinates) < 3:
                continue
                
            # 创建mask并检查点是否在其中
            mask_array = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            points = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(mask_array, [points], 255)
            
            if 0 <= y < mask_array.shape[0] and 0 <= x < mask_array.shape[1]:
                if mask_array[y, x] > 0:
                    return i
                    
        return -1
        
    def on_mask_selected(self, index: int):
        """处理mask选中事件"""
        self.selected_mask_index = index
        self.update_risk_info()
        
    def update_risk_info(self):
        """更新风险信息"""
        if self.selected_mask_index == -1:
            self.risk_info_label.setText("未选择对象")
        elif 0 <= self.selected_mask_index < len(self.current_risks):
            risk = self.current_risks[self.selected_mask_index]
            risk_desc = self.get_risk_description(risk['risk_levels'])
            
            # 获取对应的mask信息
            mask = self.current_masks[self.selected_mask_index]
            class_id = mask['class_id']
            
            # 使用类别名称（如果有）
            if self.class_names and class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class {class_id}"
            
            self.risk_info_label.setText(f"选中对象: {class_name}\n风险信息: {risk_desc}")
        
    def paintEvent(self, event):
        # 不需要手动调用image_box的paintEvent，让Qt的事件系统处理
        pass
        
    def redraw_image(self):
        """重新绘制图像"""
        if self.current_image is None:
            return
            
        # 创建显示图像
        display_image = self.create_display_image()
        
        # 转换为QPixmap
        height, width = display_image.shape[:2]
        bytes_per_line = 3 * width
        
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 更新ImageBox显示
        self.image_box.scaled_img = pixmap
        self.image_box.update()
            
    def select_mask(self, index: int):
        """通过索引选择mask
        
        Args:
            index: 要选择的mask索引，-1表示取消选择
        """
        if index == -1 or (0 <= index < len(self.current_masks)):
            self.selected_mask_index = index
            self.mask_selected.emit(index)
            self.redraw_image()


    def zoom_to_global(self):
        """缩放到全局"""
        if self.original_image is None:
            return
        
        # 计算缩放比例以适应ImageBox实际显示区域
        height, width = self.original_image.shape[:2]
        # 获取ImageBox的实际尺寸
        box_width, box_height = self.image_box.width(), self.image_box.height()
        scale_width = box_width / width
        scale_height = box_height / height
        # 使用更接近1的缩放系数，同时预留少量边距
        self.image_box.scale = min(scale_width, scale_height) * 0.95
        
        # 居中显示
        self.image_box.point = QPoint(
            int((box_width - width * self.image_box.scale) / 2),
            int((box_height - height * self.image_box.scale) / 2)
        )
        self.image_box.update()

        
    def copy_image_name(self):
        """复制图像名到剪贴板"""
        if self.current_image_name:
            from PyQt5.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(self.current_image_name)
            # 可以添加一个短暂的提示
            self.risk_info_label.setText(f"已复制文件名: {self.current_image_name}")
            import threading
            import time
            def reset_label():
                time.sleep(2)
                self.on_mask_selected(self.selected_mask_index)
            threading.Thread(target=reset_label).start()
        else:
            self.risk_info_label.setText("没有可复制的文件名")
