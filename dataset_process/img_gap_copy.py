import os
import shutil




def img_gap_copy(source_folder, destination_folder):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    file_list = sorted(os.listdir(source_folder))

    # 每10个文件选择1个进行复制
    for index, file_name in enumerate(file_list):
        if index % 10 == 0:  # 每10个选择1个
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy2(source_path, destination_path)
            print(f'Copied {file_name} to {destination_folder}')


if __name__ == '__main__':
    # 定义文件夹路径
    source_folder = r'F:\image_blurred'  # 文件夹A的路径
    destination_folder = r'E:\data\20241113_road_veg\data\src_data\image'  # 文件夹B的路径

    img_gap_copy(source_folder, destination_folder)