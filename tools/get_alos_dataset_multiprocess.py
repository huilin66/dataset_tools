# -*- coding: utf-8 -*-
# @Author  : XUNWJ
# @Contact : ssssustar@163.com
# @File    : get_alos_dataset_multiprocess.py
# @Time    : 2024/8/20 22:17
# @Desc    :
from __future__ import print_function
from __future__ import division

import multiprocessing
import os
import gc
import shutil

import numpy as np
from skimage import io
import tifffile as tff
import cv2, time
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from training_image_generator import lambertian_surface, generate_coordinates, lambertian_surface2

DATASET_PATH = r"G:/alos_dem/30m"
ALOS_SAVE_DEM1 = r'G:\alos_dem/30m_alos_dem'
ALOS_SAVE_DEM2 = r'E:\data\tp\alos_dem/30m_alos_dem'

def single_process_alos(dem_path, dem_dir):

    check_number = ''

    # region get dem
    resolution = int(os.path.basename(DATASET_PATH).split('m')[0])
    dem = io.imread(os.path.join(DATASET_PATH, dem_path))
    # dem = np.expand_dims(dem, axis=2)
    # endregion

    # region get lamb
    fake_azimuth = np.random.randint(0, 360, 2)
    elevation_angle = np.random.randint(10, 80, 2)

    light1 = np.array([int(fake_azimuth[0]), int(elevation_angle[0])], dtype=np.float32)
    light2 = np.array([int(fake_azimuth[1]), int(elevation_angle[1])], dtype=np.float32)

    # lamb1, lamb2 = lambertian_surface2(np.expand_dims(dem, axis=2), resolution, int(fake_azimuth[0]), int(elevation_angle[0]), int(fake_azimuth[1]), int(elevation_angle[1]))
    lamb1, lamb2 = lambertian_surface2(np.expand_dims(dem, axis=2), resolution, light1[0], light1[1], light2[0], light2[1])
    if np.min(lamb1) == np.max(lamb1):
        check_number = check_number + '1'
    if np.min(lamb2) == np.max(lamb2):
        check_number = check_number + '2'
    # endregion

    # region get mask
    dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem) + np.random.uniform(0.2, 0.5))
    # dem = np.squeeze(dem)
    # 计算要设置为0的像素数量
    percentage = np.random.uniform(0.95, 0.999)
    coor = generate_coordinates(percentage)
    # 将选定的像素值设置为0
    dem[coor[:, 0], coor[:, 1]] = 0.

    lamb_int = np.uint8(lamb1 * 255)
    slic = cv2.ximgproc.createSuperpixelSLIC(lamb_int, region_size=70, ruler=50)
    slic.iterate(10)
    # mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()
    num_region = np.max(label_slic) - np.min(label_slic) + 1
    # endregion

    # region get albedo
    albedo = np.ones((224, 224))
    for k in range(num_region):
        random_albedo = np.random.uniform(0.3, 0.9, 1)
        index = list(np.where(label_slic == k))
        lamb1[list(index[0]), list(index[1])] = lamb1[list(index[0]), list(index[1])] * random_albedo
        lamb2[list(index[0]), list(index[1])] = lamb2[list(index[0]), list(index[1])] * random_albedo
        albedo[list(index[0]), list(index[1])] = albedo[list(index[0]), list(index[1])] * random_albedo
    # endregion



    if '1' in check_number:
        dem_path.replace('.tif', '_resultA.tif')
    if '2' in check_number:
        dem_path.replace('.tif', '_resultB.tif')
    save_path = os.path.join(dem_dir, dem_path.replace('-%g_%g-%g_%g.tif'%
                                                             (light1[0], light1[1], light2[0], light2[1]), '.png'))

    final_image = np.vstack((np.hstack((lamb1, lamb2)), np.hstack((albedo, albedo))))
    min_val = final_image.min()
    max_val = final_image.max()
    normalized_image = (final_image - min_val) / (max_val - min_val)
    visual_image = (normalized_image * 255).astype(np.uint8)
    io.imsave(save_path, visual_image)



def multi_process_alos(dem_list, i):
    num_threads = 2*os.cpu_count()
    dem_dir = ALOS_SAVE_DEM1+'%02d'%i if i%2==1 else ALOS_SAVE_DEM2+'%02d'%i
    print(dem_dir)
    os.makedirs(dem_dir, exist_ok=True)
    with multiprocessing.Pool(processes=num_threads) as pool:
        with tqdm(dem_list) as pbar:
            single_process_alos_with_dem_dir = partial(single_process_alos, dem_dir=dem_dir)
            for _ in pool.imap_unordered(single_process_alos_with_dem_dir, dem_list):
                pbar.update()
        pool.close()
        pool.join()
    gc.collect()



if __name__ == '__main__':
    print('cpu cores:', os.cpu_count())

    dem_unsorted_list = os.listdir(DATASET_PATH)
    dem_list = sorted(dem_unsorted_list, key=lambda path: int(os.path.basename(path).split('.')[0]))
    dem_lists = np.array_split(dem_list, 10)
    for i in range(7, 8):
        print('batch',i)
        multi_process_alos(dem_lists[i], i)
        time.sleep(1)
