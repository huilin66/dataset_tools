import zipfile
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