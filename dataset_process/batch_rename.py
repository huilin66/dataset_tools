import os
import shutil


def img_copy_rename(source_folders, target_folder):


    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历所有源文件夹
    for folder in source_folders:
        # 遍历源文件夹中的所有文件
        folder_name = os.path.basename(folder)
        for filename in os.listdir(folder):
            source_file = os.path.join(folder, filename)

            # 确保它是一个文件而不是子文件夹
            if os.path.isfile(source_file):
                # 获取文件的名字和扩展名
                name, ext = os.path.splitext(filename)
                # 增加文件名后缀
                new_filename = f"{name}_{folder_name}{ext}"
                target_file = os.path.join(target_folder, new_filename)

                # 复制文件到目标文件夹
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

    print("All files have been copied successfully.")


if __name__ == '__main__':
    img_copy_rename(
        source_folders=[
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\images_vis',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo8',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo9',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\yolo10',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\mayolo',
        ],
        target_folder = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\predict\final_present\all'
    )