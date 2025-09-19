import os
import shutil
import argparse
import sys

"""本脚本用于从输入文件夹中按照文件名排序后，每隔k个文件选择一个并复制到输出文件夹。"""

def get_sorted_files(input_dir):
    """
    获取输入文件夹中的所有文件，并按照文件名排序
    
    Args:
        input_dir (str): 输入文件夹路径
    
    Returns:
        list: 排序后的文件路径列表
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹 '{input_dir}' 不存在")
        sys.exit(1)
    
    # 获取文件夹中的所有文件
    all_files = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        # 只处理文件，不处理子文件夹
        if os.path.isfile(item_path):
            all_files.append(item_path)
    
    # 按照文件名排序
    all_files.sort(key=lambda x: os.path.basename(x))
    
    return all_files

def select_files_by_gap(files, k):
    """
    按照每k个选1个的方式选择文件
    
    Args:
        files (list): 文件路径列表
        k (int): 间隔参数，每k个文件选择1个
    
    Returns:
        list: 选中的文件路径列表
    """
    # 每隔k-1个文件选择一个文件
    # 例如：k=5 时，选择索引为 0, 5, 10, ... 的文件
    selected_files = files[::k]
    return selected_files

def copy_files_to_output(selected_files, output_dir):
    """
    将选中的文件复制到输出文件夹
    
    Args:
        selected_files (list): 选中的文件路径列表
        output_dir (str): 输出文件夹路径
    """
    # 如果输出文件夹不存在，则创建它
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制文件
    copied_count = 0
    for file_path in selected_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        
        try:
            shutil.copy2(file_path, output_path)  # copy2 会保留文件的元数据
            copied_count += 1
        except Exception as e:
            print(f"复制文件 '{file_name}' 失败: {e}")
    
    print(f"已成功复制 {copied_count} 个文件到 '{output_dir}'")

def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='按照文件名排序后，每k个选1个文件复制到输出文件夹')
    parser.add_argument('--input_dir', type=str, default=r'E:\data\202502_signboard\data_annotation\id_data\select_data\labels', help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, default=r'E:\data\202502_signboard\data_annotation\id_data\select_data_select\labels', help='输出文件夹路径')
    parser.add_argument('--k', type=int, default=5, help='间隔参数，默认值为5，即每5个文件选择1个')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print(f"开始处理：从 '{args.input_dir}' 中每{args.k}个文件选择1个，复制到 '{args.output_dir}'")
    
    # 获取排序后的文件列表
    sorted_files = get_sorted_files(args.input_dir)
    print(f"在输入文件夹中找到 {len(sorted_files)} 个文件")
    
    if not sorted_files:
        print("警告: 输入文件夹中没有找到任何文件")
        return
    
    # 按照间隔k选择文件
    selected_files = select_files_by_gap(sorted_files, args.k)
    print(f"按照每{args.k}个选1个的规则，共选中 {len(selected_files)} 个文件")
    
    # 复制选中的文件到输出文件夹
    copy_files_to_output(selected_files, args.output_dir)
    
    print("处理完成！")

if __name__ == '__main__':
    main()