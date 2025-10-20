import os
import zipfile
import rarfile
import py7zr

def auto_extract(file_path, output_dir=None):
    """
    自动解压 zip, rar, 7z 文件到指定目录
    :param file_path: 压缩文件路径
    :param output_dir: 解压目标目录（默认与压缩文件同目录）
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if output_dir is None:
        output_dir = os.path.splitext(file_path)[0]  # 默认解压到同名文件夹

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔧 开始解压: {file_path} → {output_dir}")
    if ext == '.zip':
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    elif ext == '.rar':
        with rarfile.RarFile(file_path) as rar_ref:
            rar_ref.extractall(output_dir)

    elif ext == '.7z':
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            z.extractall(path=output_dir)

    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    print(f"✅ 解压完成: {file_path} → {output_dir}")
