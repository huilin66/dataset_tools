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

DATASET_PATH = r"G:\alos_dem\30m"
ALOS_SAVE_DEM = r'E:\cp_dir\alos_dem_check'
# ALOS_SAVE_DEM2 = r'E:\data\tp\alos_dem/30m_alos_dem'

def get_normal(dem, resolution):
    #compute surface normal based on 2x2 grid
    p = dem[:,1:]-dem[:,0:-1] ; p = (0.5*p[0:-1,:]+0.5*p[1:,:])/ resolution
    q = dem[1:,:]-dem[0:-1,:]; q = (0.5*q[:,0:-1]+0.5*q[:,1:])/ resolution
    return p, q

def generate_coordinates(percentage):
    # 计算图片总像素数
    total_pixels = 50176

    # 计算要生成的坐标点数
    num_coordinates = int(total_pixels * percentage)

    # 创建网格
    grid_x, grid_y = np.meshgrid(range(224), range(224))

    # 将网格坐标转为一维数组
    coordinates = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # 随机洗牌
    np.random.shuffle(coordinates)

    # 选择指定数量的坐标
    selected_coordinates = coordinates[:num_coordinates]

    return selected_coordinates

def lambertian_surface2(dem, resolution, azimuth1, elevation_angle1, azimuth2, elevation_angle2):
    illumination_vec1 =  np.matrix(get_illumination_vector_real(azimuth1, elevation_angle1))
    illumination_vec2 =  np.matrix(get_illumination_vector_real(azimuth2, elevation_angle2))

    zero = np.zeros((dem.shape[0], dem.shape[1], 1))
    one = np.ones((dem.shape[0], dem.shape[1], 1))

    gx, gy = get_normal(dem, resolution)
    gx = np.pad(gx, ((0, 1), (0, 1), (0, 0)), 'constant')
    gy = np.pad(gy, ((0, 1), (0, 1), (0, 0)), 'constant')

    # gx =  cv2.Sobel(dem,-1,1,0,ksize=3)
    gx = np.reshape(gx, [gx.shape[0], gx.shape[1], 1])
    vec_x = np.concatenate((one, zero, gx), axis=2)

    # gy =  cv2.Sobel(dem,-1,0,1,ksize=3)
    # shape_gy = np.array(np.shape(gy))
    gy = np.reshape(gy, [gy.shape[0], gy.shape[1], 1])
    vec_y = np.concatenate((zero, one, gy), axis=2)

    surface_normal = np.cross(vec_x,vec_y)
    # angle_difference = np.zeros([dem.shape[0], dem.shape[1]])
    angle_difference1 = np.array(np.matmul(surface_normal, illumination_vec1)/ (np.linalg.norm(illumination_vec1, ord = 2) * np.linalg.norm(surface_normal, ord = 2, axis = 2)))
    angle_difference2 = np.array(np.matmul(surface_normal, illumination_vec2)/ (np.linalg.norm(illumination_vec2, ord = 2) * np.linalg.norm(surface_normal, ord = 2, axis = 2)))
    return angle_difference1, angle_difference2

def get_illumination_vector_real(azimuth,elevation): #get illumination/viewing vector from azimuth and zenith angles in degrees
    theta = np.radians(azimuth); phi = np.radians(elevation)
    c_azi, s_azi = np.cos(theta), np.sin(theta)
    c_ele, s_ele = np.cos(phi), np.sin(phi)
    R_illu = np.array(((c_azi, -s_azi, 0), (s_azi, c_azi, 0),(0, 0, 1.0)))
    illumination_vector = np.matmul(R_illu,np.array([[0],[-c_ele],[s_ele]]))
    return illumination_vector

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
    save_path = os.path.join(dem_dir, dem_path.replace('.tif', '-%g_%g-%g_%g.png'%(light1[0], light1[1], light2[0], light2[1])))

    final_image = np.vstack((np.hstack((lamb1, lamb2)), np.hstack((albedo, albedo))))
    min_val = final_image.min()
    max_val = final_image.max()
    normalized_image = (final_image - min_val) / (max_val - min_val)
    visual_image = (normalized_image * 255).astype(np.uint8)
    io.imsave(save_path, visual_image)

def multi_process_alos(dem_list, i):
    num_threads = os.cpu_count()
    # dem_dir = ALOS_SAVE_DEM1+'%02d'%i if i%2==1 else ALOS_SAVE_DEM2+'%02d'%i
    dem_dir = os.path.join(ALOS_SAVE_DEM, '%02d'%i)
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
    for i in range(2, 7):
        print('batch', i)
        multi_process_alos(dem_lists[i], i)
        time.sleep(1)
