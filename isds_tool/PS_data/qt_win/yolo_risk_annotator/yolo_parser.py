#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO标签解析器
支持解析和保存包含risk信息的YOLO实例分割标签文件
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional


class YOLOParser:
    """YOLO标签解析器"""
    
    def __init__(self):
        self.risk_names = ["Risk1", "Risk2", "Risk3", "Risk4"]
        self.risk_levels = {0: "No", 1: "Medium", 2: "High"}
        
    def parse_label_file(self, label_path: str, image_shape: Tuple[int, int, int]) -> Tuple[List[Dict], List[Dict]]:
        """
        解析YOLO标签文件
        
        Args:
            label_path: 标签文件路径
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            masks: 包含mask信息的字典列表
            risks: 包含risk信息的字典列表
        """
        masks = []
        risks = []
        
        if not os.path.exists(label_path):
            return masks, risks

        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 7:  # 至少需要class_id, risk_num, 4个risk值, 至少1个坐标点
                    continue
                    
                try:
                    # 解析基本信息
                    class_id = int(parts[0])
                    risk_num = int(parts[1])
                    risk_start = 2
                    coord_start = 6  # class_id + risk_num + 4个risk值
                    if len(parts[coord_start:]) % 2 == 0:
                        coord_end = len(parts)
                        object_id = 0
                    else:
                        coord_end = len(parts) - 1
                        object_id = int(parts[-1])


                    # 解析risk值
                    risk_levels = []
                    for i in range(4):  # 固定4个risk
                        if i < len(parts) - 2:
                            risk_levels.append(int(parts[risk_start + i]))
                        else:
                            risk_levels.append(0)
                    
                    # 解析坐标点
                    coordinates = []
                    for i in range(coord_start, coord_end, 2):
                        if i + 1 < len(parts):
                            x = float(parts[i])  # 保持归一化坐标
                            y = float(parts[i + 1])
                            coordinates.append((x, y))
                    
                    # 创建mask信息
                    mask_info = {
                        'class_id': class_id,
                        'object_id': object_id,
                        'coordinates': coordinates
                    }
                    
                    # 创建risk信息
                    risk_info = {
                        'risk_num': risk_num,
                        'risk_levels': risk_levels
                    }
                    
                    masks.append(mask_info)
                    risks.append(risk_info)
                    
                except (ValueError, IndexError) as e:
                    print(f"解析标签行时出错: {line}, 错误: {e}")
                    continue
                    
        return masks, risks
        
    def save_label_file(self, label_path: str, masks: List[Dict], risks: List[Dict]) -> bool:
        """
        保存YOLO标签文件
        
        Args:
            label_path: 标签文件路径
            masks: mask信息列表
            risks: risk信息列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            with open(label_path, 'w', encoding='utf-8') as f:
                for mask, risk in zip(masks, risks):
                    # 写入class_id
                    f.write(f"{mask['class_id']} ")
                    # 写入risk_num，如果不存在则默认为0
                    risk_num = risk.get('risk_num', 4)
                    f.write(f"{risk_num} ")
                    
                    # 写入risk值
                    for level in risk['risk_levels']:
                        f.write(f"{level} ")
                    
                    # 写入坐标点（归一化到0-1）
                    coordinates = mask['coordinates']
                    if coordinates:
                        
                        # 这里假设图像尺寸，实际应该传入图像尺寸
                        # 为了简化，我们假设坐标已经是归一化的
                        for x, y in coordinates:
                            f.write(f"{x:.6f} {y:.6f} ")
                    f.write(f"{mask['object_id']} ")
                    f.write('\n')
                    
            return True
            
        except Exception as e:
            print(f"保存标签文件时出错: {e}")
            return False
            
    def create_empty_label_file(self, label_path: str) -> bool:
        """
        创建空的标签文件
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            bool: 创建是否成功
        """
        try:
            with open(label_path, 'w', encoding='utf-8') as f:
                pass
            return True
        except Exception as e:
            print(f"创建标签文件时出错: {e}")
            return False
            
    def validate_risk_levels(self, risk_levels: List[int]) -> bool:
        """
        验证risk级别是否有效
        
        Args:
            risk_levels: risk级别列表
            
        Returns:
            bool: 是否有效
        """
        if len(risk_levels) != 4:
            return False
            
        for level in risk_levels:
            if level not in [0, 1, 2]:
                return False
                
        return True
        
    def get_risk_description(self, risk_levels: List[int]) -> str:
        """
        获取risk描述
        
        Args:
            risk_levels: risk级别列表
            
        Returns:
            str: risk描述
        """
        descriptions = []
        for i, level in enumerate(risk_levels):
            if level > 0:
                level_name = self.risk_levels[level]
                descriptions.append(f"{self.risk_names[i]}: {level_name}")
                
        return ", ".join(descriptions) if descriptions else "无风险"