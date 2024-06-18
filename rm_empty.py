import os


def delete_empty_folders(folder_path):
    # 检查文件夹是否为空
    if not os.listdir(folder_path):
        os.rmdir(folder_path)
        print(f"Deleted empty folder: {folder_path}")

if __name__ == "__main__":
    pass
    root_dir = r'E:\data\0111_testdata\data_clip\img_w_result_vis_crop'
    for sub_dir in os.listdir(root_dir):
        sub_dir_path = os.path.join(root_dir, sub_dir)
        delete_empty_folders(sub_dir_path)