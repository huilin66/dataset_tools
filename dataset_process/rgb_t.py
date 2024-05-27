import os
from skimage import io
import numpy as np
from tqdm import tqdm


def get_rgbt(rgb_dir, t_dir, rgbt_dir):
    pass
    file_list = os.listdir(rgb_dir)
    for file_name in tqdm(file_list):
        rgb_path = os.path.join(rgb_dir, file_name)
        t_path = os.path.join(t_dir, file_name)
        rgbt_path = os.path.join(rgbt_dir, file_name.replace('.jpg', '.tif').replace('.png', '.tif'))

        rgb_img = io.imread(rgb_path)
        t_img = io.imread(t_path)
        assert rgb_img.shape[:2] == t_img.shape[:2]
        rgbt_img = np.concatenate((rgb_img, t_img), axis=2)
        io.imsave(rgbt_path, rgbt_img)

if __name__ == '__main__':
    pass
    # get_rgbt(rgb_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbt\rgb',
    #          t_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbt\t',
    #          rgbt_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbt\images'
    #          )

    # get_rgbt(rgb_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\rgb',
    #          t_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\tc',
    #          rgbt_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc\images'
    #          )
    # get_rgbt(rgb_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\rgb',
    #          t_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\tc_correct',
    #          rgbt_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\images'
    #          )