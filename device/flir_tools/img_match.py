import os
import shutil
import cv2
import fnv
import fnv.file
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from PIL import Image

def dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path)

def iron2gray(input_dir, output_dir):
    output_dir_norm = output_dir+'_norm'
    output_dir_color = output_dir + '_color'

    dir_check(output_dir)
    dir_check(output_dir_norm)
    dir_check(output_dir_color)

    file_list = os.listdir(input_dir)
    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.jpg', '.png'))
        output_norm_path = os.path.join(output_dir_norm, file_name.replace('.jpg', '.png'))
        output_color_path = os.path.join(output_dir_color, file_name.replace('.jpg', '.png'))

        im = fnv.file.ImagerFile(input_path)
        if not im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
            continue
        im.unit = fnv.Unit.TEMPERATURE_FACTORY
        im.temp_type = fnv.TempType.CELSIUS
        im.get_frame(0)

        data = np.array(im.final, copy=False, dtype=np.uint8).reshape((im.height, im.width))
        data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        io.imsave(output_path, data)
        io.imsave(output_norm_path, data_norm)
        shutil.copy(input_path, output_color_path)

def rgb_clip(input_dir, output_dir):
    output_dir_src = output_dir + '_src'

    dir_check(output_dir)
    dir_check(output_dir_src)

    file_list = os.listdir(input_dir)
    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.jpg', '.png'))
        output_src_path = os.path.join(output_dir_src, file_name.replace('.jpg', '.png'))
        im = fnv.file.ImagerFile(input_path)
        if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
           continue
        im.get_frame(0)
        img = np.array(im.final, copy=False, dtype=np.uint8).reshape((im.height, im.width, 3))
        img = img[172:-230, 280:-240]
        io.imsave(output_src_path, img)
        img = cv2.resize(img, (640, 480))
        io.imsave(output_path, img)


def norm_vis(input_dir1, input_dir2, output_dir):
    dir_check(output_dir)

    file_list = os.listdir(input_dir1)
    for file_name in file_list:
        input_path1 = os.path.join(input_dir1, file_name)
        input_path2 = os.path.join(input_dir2, file_name)
        output_path = os.path.join(output_dir, file_name)
        img1 = Image.open(input_path1)
        img2 = Image.open(input_path2)
        blended_image = Image.blend(img1, img2, alpha = 0.5)
        plt.imshow(blended_image)
        plt.show()


def get_pure_thermal_img(input_path, output_path):
    pass

    im = fnv.file.ImagerFile(input_path)  # open the file
    print(im.height, im.width)

    if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
        im.unit = fnv.Unit.TEMPERATURE_FACTORY
        im.temp_type = fnv.TempType.CELSIUS
    else:
        im.unit = fnv.Unit.COUNTS

    print(dir(im))

    im.get_frame(0)
    # info = im.frame_info
    # print(im.frame_info)
    data = np.array(im.final, copy=False, dtype=np.uint8).reshape((im.height, im.width))

    # data2 = np.array(im.original, copy=False).reshape((im.height, im.width))
    data2 = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # data = im.final
    # data2 = im.original
    # print()

    # plt.clf()
    # plt.imshow(data,
    #            cmap="inferno"
    #            # cmap="afmhot"
    #            # cmap="hot"
    #            # cmap="CMRmap",  # choose a color palette
    #            # aspect="auto"
    #            )  # set aspect ratio
    # plt.imshow(data)
    # plt.show()
    io.imsave(output_path, data)
    io.imsave(output_path.replace('.jpg', '_normal.jpg'), data2)

    # smap = im.subframe_map
    # print(smap)

    # no_extract = fnv.file.ImagerFileExtractOptions(preferredFormat='Image')
    # im.extract(output_path, no_extract)

    # imw = fnv.file.ImagerFileWriter(output_path,
    #                                 im.reduce_objects,
    #                                 flags=fnv.file.ImagerFileFlags.SINGLE_PRESET,
    #                                 # preferredFormat='TIFF'
    #                                 )
    # r = imw.put_frame(im.original)
    # print(r)
    # fs = imw.get_formats()
    # print(fs)
    # # imw.create(output_path, im.reduce_objects, preferredFormat='TIFF')
    # imw.close()

def rgb_clip_single(input_path, output_path):
    img = io.imread(input_path)
    print(img.shape)
    img = img[172:-230, 280:-240]
    print(img.shape)
    # plt.imshow(img)
    # plt.show()

    # 调整透明度
    img = cv2.resize(img, (640, 480))
    io.imsave(output_path, img)
    img = Image.fromarray(img)
    img2 = Image.open(r'E:\data\0417_signboard\data0507\demo\FLIR0735.jpg')
    blended_image = Image.blend(img, img2, alpha = 0.5)

    plt.imshow(blended_image)
    plt.show()



if __name__ == '__main__':
    pass
    # rgb_clip_single(input_path=r'E:\data\0417_signboard\data0507\demo\FLIR0736.jpg',
    #                 output_path=r'E:\data\0417_signboard\data0507\demo\FLIR0736_clip.jpg')

    # iron2gray(input_dir=r'E:\data\0417_signboard\data0507\src',
    #           output_dir=r'E:\data\0417_signboard\data0507\norm\thermal')
    #
    # rgb_clip(input_dir=r'E:\data\0417_signboard\data0507\src',
    #          output_dir=r'E:\data\0417_signboard\data0507\norm\rgb')

    # norm_vis(input_dir1, input_dir2, output_dir)


    # iron2gray(input_dir=r'E:\data\0417_signboard\data0514\src',
    #           output_dir=r'E:\data\0417_signboard\data0514\norm\thermal')

    # rgb_clip(input_dir=r'E:\data\0417_signboard\data0514\src',
    #          output_dir=r'E:\data\0417_signboard\data0514\norm\rgb')