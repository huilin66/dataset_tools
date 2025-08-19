import pandas as pd
import os
import shutil
import argparse
from pathlib import Path
from yolo2xanylabeling import yolo_to_xanylabeling_dir
def copy_files_from_excel(excel_path, source_image_dir, target_image_dir, source_label_dir, target_label_dir):
    # 读取Excel文件
    try:
        df = pd.read_excel(excel_path)
        # 假设Excel中包含文件名的列名为'filename'，如果不是，可以修改这里
        filenames = df.iloc[:, 0].tolist()  # 取第一列数据
        print(f"从Excel中读取到{len(filenames)}个文件名")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 确保目标目录存在
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0
    missing_files = []


    img_list = os.listdir(source_image_dir)
    img_stem_list = [Path(img).stem for img in img_list]
    stem2img_dict = dict(zip(img_stem_list, img_list))

    # 复制文件
    for filename in filenames:
        # 处理可能的扩展名问题
        base_name = Path(filename).stem
        if base_name not in stem2img_dict:
            missing_count += 1
            continue
        img_name = stem2img_dict[base_name]
        src_img_path = os.path.join(source_image_dir, img_name)

        label_filename = base_name + '.txt'
        src_label_path = os.path.join(source_label_dir, label_filename)

        # 复制图像文件
        dst_img_path = os.path.join(target_image_dir, filename)
        shutil.copy2(src_img_path, dst_img_path)
        print(f"复制图像文件: {filename} -> {target_image_dir}")

        # 复制标签文件(如果存在)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(target_label_dir, label_filename)
            shutil.copy2(src_label_path, dst_label_path)
            print(f"复制标签文件: {label_filename} -> {target_label_dir}")
        else:
            print(f"警告: 未找到标签文件 {label_filename}")

        copied_count += 1

    # 打印结果
    print(f"复制完成: 成功复制{copied_count}个文件，缺失{missing_count}个文件")
    if missing_count > 0:
        print("缺失的文件:")
        for file in missing_files:
            print(f"  - {file}")

if __name__ == "__main__":

    excel_file = r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\image_error.xlsx'
    source_img = r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3899_mseg_c6_0818\images'
    target_img=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\select\images'
    source_label=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\data3899_mseg_c6_0818\labels'
    target_label=r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\select\labels'
    # copy_files_from_excel(
    #     excel_file,
    #     source_img,
    #     target_img,
    #     source_label,
    #     target_label
    # )

    class_file = r'E:\data\202502_signboard\data_annotation\docs\class_c6.txt'
    attribute_file = r'E:\data\202502_signboard\data_annotation\docs\attribute.yaml'

    xanylabeling_labeing_dir = r'E:\data\202502_signboard\data_annotation\ps_data\dataset_result\select\json'

    # img2png(images_dir, images_re_dir)
    yolo_to_xanylabeling_dir(target_label, target_img, xanylabeling_labeing_dir, class_file, attribute_file)