import os
import shutil
import argparse

def move_labels(source_dir):
    # 定义目标文件夹路径
    labels_dir = os.path.join(source_dir, 'labels')
    
    # 创建labels文件夹（如果不存在）
    os.makedirs(labels_dir, exist_ok=True)
    
    # 定义图像文件扩展名
    image_extensions = ['.txt']
    
    # 统计移动的文件数量
    moved_count = 0
    
    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(source_dir):
        # 获取文件扩展名
        _, ext = os.path.splitext(file_name)
        
        # 检查是否为图像文件
        if ext.lower() in image_extensions:
            # 构建完整的文件路径
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(labels_dir, file_name)
            
            # 移动文件
            shutil.move(source_path, target_path)
            print(f"已移动: {file_name}")
            moved_count += 1
    
    print(f"移动完成，共移动了 {moved_count} 个图像文件到 {labels_dir}")