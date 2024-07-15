import shutil
import numpy as np
import py360convert
from skimage import io
import os
from tqdm import tqdm
from equirectRotate import EquirectRotate, pointRotate
cube_names = ['F', 'R', 'B', 'L', 'U', 'D']

# 'F', 'R', 'B', 'L', 'U', 'D'
def p2c(panorama_path, save_dir, erot=None):
    cube_dice = io.imread(panorama_path)

    if erot is not None:
        cube_dice = erot.rotate(cube_dice)

    cubemap = py360convert.e2c(cube_dice, face_w=2048, cube_format = 'list')
    for i,cube in enumerate(cubemap):
        if i == 5:
            continue
        if i == 1 or i == 2:
            cube = np.flip(cube, axis=1)
        elif i == 4:
            cube = np.flip(cube, axis=0)
        save_path = os.path.join(save_dir, '%s_%s.png'%(os.path.basename(panorama_path).replace('.jpg','').replace('.png',''), cube_names[i]))
        io.imsave(save_path, cube)
        print('save to ', save_path)


def data_group(input_dir, output_dir):
    for cube_name in cube_names:
        cube_path = os.path.join(output_dir, cube_name)
        if not os.path.exists(cube_path):
            os.makedirs(cube_path)

    for file_name in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, '%s'%file_name[-5:-4], file_name)
        shutil.move(input_path, output_path)

if __name__ == '__main__':
    pass
    # panorama_dir = r'E:\data\0417_signboard\data_yj\imgs_board'
    # panorama_dir = r'E:\data\0417_signboard\data0417\yolo\images'
    # panorama_path = r'E:\data\0417_signboard\data0420\yolo\images\1702263237.8748906.png'
    # save_dir = r'E:\data\0417_signboard\data_yj\imgs_board_cube'
    # save_dir_group = r'E:\data\0417_signboard\data_yj\imgs_board_cube_group2'
    panorama_dir = r'E:\data\0417_signboard\VMMS\ladybug\4\panoramic'
    save_dir = r'E:\data\0417_signboard\VMMS\ladybug\4\cube'
    save_dir_group = r'E:\data\0417_signboard\VMMS\ladybug\4\cube_group'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_group, exist_ok=True)

    # equirectRot = EquirectRotate(1920, 3840, (0, 0, 20))
    equirectRot = None

    for file_name in tqdm(os.listdir(panorama_dir)[::]):
        panorama_path = os.path.join(panorama_dir, file_name)
        p2c(panorama_path, save_dir, equirectRot)

    data_group(save_dir, save_dir_group)

