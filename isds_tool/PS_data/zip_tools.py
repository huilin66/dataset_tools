import zipfile
import subprocess
import os

def zip_folder_to_path(source_folder, destination_zip):
    with zipfile.ZipFile(destination_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # 在zip文件中创建相对路径
                arcname = os.path.relpath(file_path, start=source_folder)
                zipf.write(file_path, arcname)
    
    print(f"zip '{source_folder}' to '{destination_zip}'")


def uzip_dirs(root_dir, zip_relative_path):
    sub_dir_list = os.listdir(root_dir)
    for sub_dir_name in sub_dir_list:
        sub_dir = os.path.join(root_dir, sub_dir_name)
        zip_path = os.path.join(sub_dir, zip_relative_path)
        if not os.path.exists(zip_path):
            print(f'{zip_path} not exists')
        else:
            print(f'{zip_path} unzip...')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_path.replace('.zip', ''))
            print(f'{zip_path} done\n')


def uzip_file(zip_path, result_path):
    if not os.path.exists(zip_path):
        print(f'{zip_path} not exists')
    else:
        print(f'{zip_path} unzip...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(result_path)
        print(f'{zip_path} done\n')



def uzip_fastest(zip_path, result_path):
    print(f'{zip_path} unzip with 7z (Subprocess)...')
    os.makedirs(result_path, exist_ok=True)
    
    # 构造命令: 7z x "source.zip" -o"dest_folder" -y
    # x: 解压并保持目录结构
    # -o: 指定输出目录 (注意-o后面紧跟路径，没有空格)
    # -y: 自动覆盖不提示
    # -bsp1: (可选) 输出进度到控制台
    cmd = ['7z', 'x', zip_path, f'-o{result_path}', '-y', '-bsp1']
    
    try:
        # 调用命令行
        subprocess.run(cmd, check=True)
        print(f'\n{zip_path} done')
    except FileNotFoundError:
        print("Error: 7z command not found. Please install 7-Zip.")
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")

if __name__ == '__main__':
    source_folder = r'\\158.132.186.40\isds\huilin\coco\coco.zip'
    result_path = r'\\158.132.186.40\isds\huilin\coco\coco'
    uzip_fastest(source_folder, result_path)