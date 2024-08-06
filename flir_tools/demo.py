import os
import cv2
import fnv.file
import shutil
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image

X1, X2, Y1, Y2 = 172,-230, 280,-240

def images_split(input_dir, rgb_dir, t_dir):
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(t_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_file = os.path.join(input_dir, file_name)
        rgb_file = os.path.join(rgb_dir, file_name)
        t_file = os.path.join(t_dir, file_name)
        file_num = int(file_name[4:8])
        if file_num % 2 == 1:
            shutil.copyfile(input_file, t_file)
        else:
            shutil.copyfile(input_file, rgb_file)

def imgt_rename(t_dir):
    def name_t2rgb(file_name):
        idx_t = int(file_name[4:8])
        idx_rgb = idx_t + 1
        t_name = file_name[:4] + '%04d' % (idx_rgb) + file_name[8:]
        return t_name.replace('.jpg', '.png')
    file_list = os.listdir(t_dir)
    for file_name in tqdm(file_list):
        src_path = os.path.join(t_dir, file_name)
        dst_path = os.path.join(t_dir, name_t2rgb(file_name))
        os.rename(src_path, dst_path)


def rgb_clip(rgb_dir, rgb_clip_dir):
    os.makedirs(rgb_clip_dir, exist_ok=True)

    file_list = os.listdir(rgb_dir)
    for file_name in tqdm(file_list):
        rgb_path = os.path.join(rgb_dir, file_name)
        rgb_clip_path = os.path.join(rgb_clip_dir, file_name.replace('.jpg', '.png'))

        im = fnv.file.ImagerFile(rgb_path)

        im.get_frame(0)
        img = np.array(im.final, copy=False, dtype=np.uint8).reshape((im.height, im.width, 3))
        img_clip = img[X1:X2, Y1:Y2]
        img_clip = cv2.resize(img_clip, (640, 480))
        io.imsave(rgb_clip_path, img_clip)

def rgbt_merge(rgb_dir, t_dir, rgbt_dir):
    os.makedirs(rgbt_dir, exist_ok=True)

    file_list = os.listdir(rgb_dir)
    for file_name in tqdm(file_list):
        rgb_path = os.path.join(rgb_dir, file_name)
        t_path = os.path.join(t_dir, file_name)
        rtbt_path = os.path.join(rgbt_dir, file_name)
        img_rgb = Image.open(rgb_path)
        img_t = Image.open(t_path)
        img_rgbt = Image.blend(img_rgb, img_t, alpha = 0.5)
        img_rgbt.save(rtbt_path)

if __name__ == '__main__':
    pass
    # images_split(
    #     input_dir=r'E:\data\0417_signboard\data0806\src\images',
    #     rgb_dir=r'E:\data\0417_signboard\data0806\src\rgb',
    #     t_dir=r'E:\data\0417_signboard\data0806\src\t'
    # )
    # imgt_rename(t_dir=r'E:\data\0417_signboard\data0806\src\t')
    # rgb_clip(
    #     rgb_dir=r'E:\data\0417_signboard\data0806\src\rgb',
    #     rgb_clip_dir=r'E:\data\0417_signboard\data0806\src\rgb_clip',
    # )
    rgbt_merge(
        rgb_dir=r'E:\data\0417_signboard\data0806\src\rgb_clip',
        t_dir=r'E:\data\0417_signboard\data0806\src\t',
        rgbt_dir=r'E:\data\0417_signboard\data0806\src\rgbt',
    )