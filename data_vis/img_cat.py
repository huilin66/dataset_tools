import os
import cv2
import numpy as np
from pathlib import Path

def horizontal_concatenate_images(folder1, folder2, output_folder, allowed_extensions=None):
    """
    将两个文件夹中的同名图像水平拼接
    
    Args:
        folder1 (str): 第一个文件夹路径
        folder2 (str): 第二个文件夹路径
        output_folder (str): 输出文件夹路径
        allowed_extensions (list): 允许的图像扩展名列表，默认为常见图像格式
    """
    # 设置默认允许的图像扩展名
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取两个文件夹中的所有文件
    files1 = {f.name: f.path for f in os.scandir(folder1) if f.is_file()}
    files2 = {f.name: f.path for f in os.scandir(folder2) if f.is_file()}
    
    # 找到两个文件夹中同名的文件（忽略扩展名）
    # 按文件名（不含扩展名）进行匹配
    names1 = {}
    for name, path in files1.items():
        base_name, ext = os.path.splitext(name)
        if ext.lower() in allowed_extensions:
            names1[base_name] = (name, path)
    
    names2 = {}
    for name, path in files2.items():
        base_name, ext = os.path.splitext(name)
        if ext.lower() in allowed_extensions:
            names2[base_name] = (name, path)
    
    # 找到同名的文件
    common_names = set(names1.keys()) & set(names2.keys())
    
    if not common_names:
        print("警告：两个文件夹中没有找到同名图像文件！")
        return
    
    print(f"找到 {len(common_names)} 对同名图像文件")
    
    success_count = 0
    failed_files = []
    
    for base_name in common_names:
        name1, path1 = names1[base_name]
        name2, path2 = names2[base_name]
        
        try:
            # 读取图像
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None:
                raise ValueError(f"无法读取图像: {path1}")
            if img2 is None:
                raise ValueError(f"无法读取图像: {path2}")
            
            # 调整图像高度一致（以较小的高度为准）
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            if h1 != h2:
                # 如果高度不同，调整到相同高度
                target_height = min(h1, h2)
                if h1 > target_height:
                    img1 = cv2.resize(img1, (int(w1 * target_height / h1), target_height))
                if h2 > target_height:
                    img2 = cv2.resize(img2, (int(w2 * target_height / h2), target_height))
            
            # 水平拼接图像
            concatenated = np.hstack((img1, img2))
            
            # 保存拼接后的图像
            # 使用第一个图像的扩展名
            _, ext = os.path.splitext(name1)
            output_path = os.path.join(output_folder, f"{base_name}_concatenated{ext}")
            
            cv2.imwrite(output_path, concatenated)
            success_count += 1
            print(f"✓ 已拼接: {base_name}")
            
        except Exception as e:
            failed_files.append((base_name, str(e)))
            print(f"✗ 处理失败 {base_name}: {e}")
    
    # 输出统计信息
    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"成功拼接: {success_count} 对图像")
    print(f"失败: {len(failed_files)} 对图像")
    if failed_files:
        print("失败的文件:")
        for name, error in failed_files:
            print(f"  - {name}: {error}")
    print(f"输出文件夹: {output_folder}")
    print("="*50)


def main(folder1, folder2, output_folder):
    """主函数 - 演示如何使用"""
    # 获取用户输入
    print("图像水平拼接工具")
    print("="*30)
    

    # 检查文件夹是否存在
    if not os.path.exists(folder1):
        print(f"错误：文件夹 '{folder1}' 不存在！")
        return
    if not os.path.exists(folder2):
        print(f"错误：文件夹 '{folder2}' 不存在！")
        return
    
    # 执行拼接
    horizontal_concatenate_images(folder1, folder2, output_folder)


if __name__ == "__main__":
    # val_input_dir = r'\\158.132.186.40\isds\huilin\tp\eccv_dn\Drop'
    # val_infer_dir = r'\\158.132.186.40\isds\huilin\tp\eccv_dn\jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260708_101616'
    # val_cat_dir = r'E:\demo\rain_drop\val_cat'

    # main(val_input_dir, val_infer_dir, val_cat_dir)    
    
    test_input_dir = r'\\158.132.186.40\isds\huilin\tp\eccv_dn\test-input'
    test_infer_dir = r'\\158.132.186.40\isds\huilin\tp\eccv_dn\jit_submit_best_model_ema1_s1_r16_hflip_rot90_best_20260710_151811'
    test_cat_dir = r'E:\demo\rain_drop\test_cat'

    main(test_input_dir, test_infer_dir, test_cat_dir)