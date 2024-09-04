import os

import cv2
from tqdm import tqdm

def name_rgb2t(file_name):
    #FLIR0893.jpg
    idx_rgb = int(file_name[4:8])
    idx_t = idx_rgb - 1
    t_name = file_name[:4]+'%04d'%(idx_t)+file_name[8:]
    return t_name

def copy_t_by_rgb(rgb_dir, t_dir, dst_dir):
    pass
    file_list = os.listdir(rgb_dir)
    for file_name in tqdm(file_list):
        t_path = os.path.join(t_dir, name_rgb2t(file_name))
        dst_path = os.path.join(dst_dir, file_name)
        img = cv2.imread(t_path)
        img = cv2.resize(img, (640, 480))
        cv2.imwrite(dst_path, img)

if __name__ == '__main__':
    pass
    # rgb_dir = r'E:\data\0417_signboard\data0521_m\yolo_t\images'
    # t_dir = r'E:\data\0417_signboard\data0521_m\norm\thermal'
    # copy_t_by_rgb(rgb_dir, t_dir, rgb_dir)

    # rgb_dir = r'E:\data\0417_signboard\data0521_m\yolo_tn\images'
    # t_dir = r'E:\data\0417_signboard\data0521_m\norm\thermal_norm'
    # copy_t_by_rgb(rgb_dir, t_dir, rgb_dir)

    # rgb_dir = r'E:\data\0417_signboard\data0521_m\yolo_tc\images'
    # t_dir = r'E:\data\0417_signboard\data0521_m\norm\thermal_color'
    # copy_t_by_rgb(rgb_dir, t_dir, rgb_dir)