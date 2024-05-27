import os, cv2
import numpy as np
from skimage import io
from tqdm import tqdm

file_off5_list = [
    "FLIR0732.png",

    "FLIR0796.png",
    "FLIR0808.png",
    "FLIR0834.png",
    "FLIR0858.png",
    "FLIR0894.png",
    "FLIR0896.png",
    "FLIR0898.png",
    "FLIR0900.png",
    "FLIR0908.png",
    "FLIR0992.png",
    "FLIR1044.png",

]

file_pad10_list = [
    "FLIR0752.png",
]
file_pad_10_list = [
    "FLIR0902.png",
]
file_off10_list = [



    "FLIR0766.png",


    "FLIR0812.png",
    "FLIR0816.png",
    "FLIR0830.png",

    "FLIR0850.png",

    "FLIR0860.png",
    "FLIR0862.png",
    "FLIR0864.png",



    "FLIR0910.png",
    "FLIR0916.png",
    "FLIR0940.png",
    "FLIR0942.png",
    "FLIR0972.png",
    "FLIR0974.png",

    "FLIR1028.png",
    "FLIR1030.png",

    "FLIR1048.png",
    "FLIR1050.png",
    "FLIR1056.png",
]

file_off20_pad10_list = [
    "FLIR0986.png",
]
file_pad_30_list = [
    "FLIR0982.png",
]

file_label_correct_list = [
    # 'FLIR0764.png',
    # 'FLIR0766.png',
    # 'FLIR0804.png',

    # 'FLIR0982.png',
    # 'FLIR0986.png',
    # 'FLIR1004.png',
    # 'FLIR1016.png',
    # 'FLIR1042.png',
]

def img_pad(img_path, dst_path, pad_bt=None, pad_rt=None, pad_lf=None):
    img = io.imread(img_path)
    print(img.shape)
    if pad_bt is not None:
        img_pad = img[-pad_bt:]
        img = np.concatenate((img,img_pad), axis=0)
    if pad_rt is not None:
        img_pad = img[:, -pad_rt:]
        img = np.concatenate((img, img_pad), axis=1)
    if pad_lf is not None:
        img_pad = img[:, :pad_lf]
        img = np.concatenate((img_pad, img), axis=1)
    img = cv2.resize(img, (640, 480))
    io.imsave(dst_path, img)

def img_clip(img_path, dst_path, off_tp=None, off_rt=None):
    img = io.imread(img_path)

    if off_tp is not None:
        img = img[off_tp:]
    if off_rt is not None:
        img = img[:, :-off_rt]
    img = cv2.resize(img, (640, 480))
    io.imsave(dst_path, img)

if __name__ == '__main__':
    img_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\tc'
    dst_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\tc_correct'

    for file_name in tqdm(file_pad_30_list):
        img_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        img_pad(dst_path, dst_path, pad_lf=30)


    for file_name in tqdm(file_off20_pad10_list):
        img_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        img_clip(img_path, dst_path, off_tp=30)
        img_pad(dst_path, dst_path, pad_rt=20)

    for file_name in tqdm(file_off10_list):
        img_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        img_clip(img_path, dst_path, off_tp=10)
    for file_name in tqdm(file_off5_list):
        img_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        img_clip(img_path, dst_path, off_tp=5)

    for file_name in tqdm(file_pad10_list):
        img_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        img_pad(img_path, dst_path, pad_bt=10)