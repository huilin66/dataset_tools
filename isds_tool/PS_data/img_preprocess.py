import os
import shutil
import os
from PIL import Image
import imagehash
from tqdm import tqdm

def select_img(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_list = os.listdir(input_dir)
    img_list = img_list[::10]
    for img_name in tqdm(img_list):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        shutil.copyfile(input_path, output_path)




def find_repeated_images(input_dir):
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])


    # 计算每张图的感知哈希值
    hashes = []
    for file in tqdm(image_files):
        img_path = os.path.join(img_folder, file)
        hash_val = imagehash.phash(Image.open(img_path))
        hashes.append((file, hash_val))

    # 找出相似图片（哈希距离小于等于 threshold）
    threshold = 2
    similar_groups = []

    for i in range(len(hashes)):
        group = [hashes[i][0]]
        for j in range(i + 1, len(hashes)):
            dist = abs(hashes[i][1] - hashes[j][1])
            if dist <= threshold:
                group.append(hashes[j][0])
            else:
                break  # 因为图片顺序拍摄，一旦不同就跳出
        if len(group) > 1:
            similar_groups.append(group)

    # 输出结果
    for group in similar_groups:
        print("重复组：", group)


if __name__ == '__main__':
    pass
    # input_dir = r'Y:\ZHL\isds\PS\20250616\rectified_image\rectified_image'
    # output_dir = r'E:\data\202502_signboard\PS\20250616\selected_img'
    # select_img(input_dir, output_dir)
    # 路径下所有图片
    # img_folder = r"E:\data\202502_signboard\PS\20250616\selected_img"
    # find_repeated_images(img_folder)

    # input_dir = r'E:\data\202502_signboard\PS\20250616\rectified_image2\rectified_image'
    # output_dir = r'E:\data\202502_signboard\PS\20250616\rectified_image2\selected_img'
    # select_img(input_dir, output_dir)

