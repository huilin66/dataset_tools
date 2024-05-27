import os
from PIL import Image
from tqdm import tqdm

def img_overlay(img1_path, img2_path, overlay_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    overlay = Image.blend(img1, img2, alpha=0.5)

    overlay.save(overlay_path)

def img_overlay_dir(img1_dir, img2_dir, overlay_dir):
    file_list = os.listdir(img1_dir)
    for file_name in tqdm(file_list):
        img1_path = os.path.join(img1_dir, file_name)
        img2_path = os.path.join(img2_dir, file_name)
        overlay_path = os.path.join(overlay_dir, file_name)
        img_overlay(img1_path, img2_path, overlay_path)

if __name__ == '__main__':
    # img_overlay(r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\rgb\FLIR0726.png',
    #             r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\tc\FLIR0726.png',
    #             r'FLIR0726.png')

    img_overlay_dir(r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\rgb',
                r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\tc',
                r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\overlay')