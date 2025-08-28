#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险编辑组件
用于编辑选中mask的风险信息和object_id
"""

from typing import List, Dict, Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QGridLayout,
                             QRadioButton, QButtonGroup, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal


class RiskEditorWidget(QWidget):
    """风险编辑组件"""
    
    risk_updated = pyqtSignal(dict)  # 发送更新后的风险数据
    object_id_updated = pyqtSignal(int)  # 发送更新后的object_id
    
    def __init__(self):
        super().__init__()
        self.current_risk_data = None
        self.current_object_id = None
        self.risk_groups = []  # 存储每个风险的按钮组
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建组框
        group_box = QGroupBox("风险信息与ID编辑")
        group_layout = QVBoxLayout(group_box)
        
        # Object ID输入
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Object ID:"))
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("输入数字ID")
        id_layout.addWidget(self.id_input)
        
        # ID更新按钮
        id_update_button = QPushButton("更新ID")
        id_update_button.clicked.connect(self.on_id_update_clicked)
        id_layout.addWidget(id_update_button)
        
        group_layout.addLayout(id_layout)
        
        # 风险级别设置
        self.create_risk_controls(group_layout)
        
        # 更新按钮
        update_button = QPushButton("更新风险信息")
        update_button.clicked.connect(self.on_update_clicked)
        group_layout.addWidget(update_button)
        
        layout.addWidget(group_box)
        
    def create_risk_controls(self, parent_layout):
        """创建风险控制组件"""
        # 创建网格布局
        grid_layout = QGridLayout()
        
        # 添加标题
        grid_layout.addWidget(QLabel("风险类型"), 0, 0)
        grid_layout.addWidget(QLabel("风险级别"), 0, 1)
        
        # 创建4个风险控制
        self.risk_groups = []
        risk_names = ["abandonment", "broken", "corrosion", "deformation"]
        risk_levels = ["No", "Medium", "High"]
        
        for i in range(4):
            # 风险名称标签
            name_label = QLabel(risk_names[i])
            grid_layout.addWidget(name_label, i + 1, 0)
            
            # 创建按钮组
            button_group = QButtonGroup(self)
            level_layout = QHBoxLayout()
            
            # 创建单选按钮
            for level in risk_levels:
                radio = QRadioButton(level)
                button_group.addButton(radio)
                level_layout.addWidget(radio)
                
                # 默认选择"No"
                if level == "No":
                    radio.setChecked(True)
            
            self.risk_groups.append(button_group)
            grid_layout.addLayout(level_layout, i + 1, 1)
            
        parent_layout.addLayout(grid_layout)
        
    def set_risk_data(self, risk_data: Dict, object_id: int = None):
        """
        设置风险数据和object_id
        
        Args:
            risk_data: 风险数据字典
            object_id: 对象ID
        """
        self.current_risk_data = risk_data.copy()
        self.current_object_id = object_id
        
        # 设置object_id
        if object_id is not None:
            self.id_input.setText(str(object_id))
        else:
            self.id_input.clear()
        
        # 设置风险级别
        risk_levels = risk_data.get('risk_levels', [0, 0, 0, 0])
        level_names = {0: "No", 1: "Medium", 2: "High"}
        
        for i, group in enumerate(self.risk_groups):
            if i < len(risk_levels):
                level = risk_levels[i]
                level_name = level_names.get(level, "No")
                
                # 查找并选中对应按钮
                for button in group.buttons():
                    if button.text() == level_name:
                        button.setChecked(True)
                        break
            else:
                # 默认选择"No"
                for button in group.buttons():
                    if button.text() == "No":
                        button.setChecked(True)
                        break
                
    def get_risk_data(self) -> Dict:
        """获取当前风险数据"""
        # 获取风险级别
        risk_levels = []
        level_values = {"No": 0, "Medium": 1, "High": 2}
        
        for group in self.risk_groups:
            for button in group.buttons():
                if button.isChecked():
                    level_name = button.text()
                    level_value = level_values.get(level_name, 0)
                    risk_levels.append(level_value)
                    break
                
        return {
            'risk_levels': risk_levels
        }
        
    def on_update_clicked(self):
        """更新按钮点击事件"""
        if self.current_risk_data is None:
            QMessageBox.warning(self, "警告", "请先选择一个mask")
            return
            
        # 获取当前风险数据
        new_risk_data = self.get_risk_data()
        
        # 验证数据
        if not self.validate_risk_data(new_risk_data):
            QMessageBox.warning(self, "警告", "风险数据无效")
            return
            
        # 发送更新信号
        self.risk_updated.emit(new_risk_data)
        
        QMessageBox.information(self, "成功", "风险信息已更新")
        
    def on_id_update_clicked(self):
        """更新ID按钮点击事件"""
        if self.current_object_id is None:
            QMessageBox.warning(self, "警告", "请先选择一个mask")
            return
            
        try:
            new_id = int(self.id_input.text())
            if new_id < 0:
                raise ValueError("ID必须为非负整数")
            
            # 发送ID更新信号
            self.object_id_updated.emit(new_id)
            self.current_object_id = new_id
            
            QMessageBox.information(self, "成功", f"Object ID已更新为: {new_id}")
        except ValueError as e:
            QMessageBox.warning(self, "警告", f"无效的ID: {str(e)}")
        
    def validate_risk_data(self, risk_data: Dict) -> bool:
        """验证风险数据"""
        risk_levels = risk_data.get('risk_levels', [])
        
        # 检查风险级别
        if len(risk_levels) != 4:
            return False
            
        for level in risk_levels:
            if level not in [0, 1, 2]:
                return False
                
        return True
        
    def clear(self):
        """清除数据"""
        self.current_risk_data = None
        self.current_object_id = None
        self.id_input.clear()
        
        for group in self.risk_groups:
            for button in group.buttons():
                if button.text() == "No":
                    button.setChecked(True)
                    break