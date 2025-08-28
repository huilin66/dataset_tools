import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开原图
                img = Image.open(input_path)

                # resize 到 128x128
                resized = img.resize((128, 128), Image.LANCZOS)

                # 创建 960x608 黑色背景
                background = Image.new("RGB", (960, 608), (0, 0, 0))

                # 计算居中位置
                x = (960 - 128) // 2
                y = (608 - 128) // 2

                # 粘贴
                background.paste(resized, (x, y))

                # 保存
                background.save(output_path)
                print(f"✅ 处理完成: {output_path}")

            except Exception as e:
                print(f"❌ 处理 {filename} 出错: {e}")


def infer2label(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for filename in tqdm(file_list):
        risks = filename.split('_')[2]
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        new_lines = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                new_line = f'{line[:1]} 4 {risks[0]} {risks[1]} {risks[2]} {risks[3]}{line[1:]}'
                new_lines.append(new_line)
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

def data_check(data_dir):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    count = 0
    image_list = os.listdir(image_dir)
    for image_name in tqdm(image_list):
        label_name = image_name.replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_name)
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(label_path):
            os.remove(image_path)
            count += 1
    print(f'remove {count} images')


if __name__ == '__main__':
    input_dir = r'/data/huilin/data/isds/fused_data/data3899_mseg_c6_0818/diffusion_data'
    yolo_dir = r'/data/huilin/data/isds/fused_data/diffusion_data_0821'
    output_dir = r'/data/huilin/data/isds/fused_data/diffusion_data_0821/images'
    infer_dir = r'/data/huilin/data/isds/fused_data/diffusion_data_0821/infer/labels'
    labels_dir = r'/data/huilin/data/isds/fused_data/diffusion_data_0821/labels'
    csv_path = r'/data/huilin/data/isds/fused_data/data3899_mseg_c6_0818/diffusion_data.csv'
    # process_images(input_dir, output_dir)
    infer2label(infer_dir, labels_dir)
    data_check(yolo_dir)