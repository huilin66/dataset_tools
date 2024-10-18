import os
import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

# 计算感知哈希相似度 (pHash)
def calculate_phash_similarity(imageA_path, imageB_path):
    # 打开图片并计算感知哈希值
    hashA = imagehash.phash(Image.open(imageA_path))
    hashB = imagehash.phash(Image.open(imageB_path))

    # 计算哈希值之间的差距（距离越小，图片越相似）
    return abs(hashA - hashB)

# 计算 SSIM
def calculate_ssim(imageA, imageB):
    # 将图像转换为灰度图以计算 SSIM
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

# 计算直方图相似度
def calculate_histogram_similarity(imageA, imageB):
    # 计算图片的直方图，并进行归一化
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()

    # 使用相关性比较直方图
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return similarity
import cv2
import numpy as np

def find_translation(imageA, imageB):
    # 转换为灰度图
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 检测特征点和描述符
    keypointsA, descriptorsA = orb.detectAndCompute(grayA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(grayB, None)

    # 使用BFMatcher进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptorsA, descriptorsB)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配点
    pointsA = np.float32([keypointsA[m.queryIdx].pt for m in matches])
    pointsB = np.float32([keypointsB[m.trainIdx].pt for m in matches])

    # 使用RANSAC剔除错误匹配，并估计平移矩阵
    if len(pointsA) >= 400:  # 至少需要4个点来计算仿射变换
        matrix, mask = cv2.estimateAffinePartial2D(pointsA, pointsB, method=cv2.RANSAC)
        if matrix is not None:
            # 平移量是仿射变换矩阵中的最后一列
            translation = matrix[:, 2]
            return translation, mask.sum()  # 返回平移量和有效匹配的数量
        else:
            return None, 0
    else:
        return None, 0


def phase_correlation(imageA, imageB):
    # 将图片转换为灰度图
    r1, g1, b1 = cv2.split(imageA)
    r2, g2, b2 = cv2.split(imageB)

    # 使用傅里叶变换计算相位相关
    shift1, p1 = cv2.phaseCorrelate(np.float32(r1), np.float32(r2))
    shift2, p2 = cv2.phaseCorrelate(np.float32(g1), np.float32(g2))
    shift3, p3 = cv2.phaseCorrelate(np.float32(b1), np.float32(b2))
    shift = shift1 + shift2 + shift3
    p = p1 + p2 + p3

    return shift, p

# # 图片路径
# img_dir = r'E:\data\2024_defect\2024_defect_pure_yolo\8.10-train-e3pt6-rfruo\train\images'
# # img_name1 = r'IMG_0760_JPG_jpg.rf.ea590674028b55c5af4b404ffbdc7d7d.jpg'
# # img_name2 = r'IMG_0761_JPG_jpg.rf.5bf795b640168089ed03d49e0dbbf1c2.jpg'
# # img_name2 = r'IMG_0763_JPG_jpg.rf.493a14f4da27268dae4f707dfe380211.jpg'
# '''
# SSIM: 0.8175189262868251
# Histogram Similarity: 0.7842843485090284
# pHash similarity (distance): 8
# SSIM: 0.76299829869551
# Histogram Similarity: 0.6590532336908695
# pHash similarity (distance): 24
# '''
# # img_name1 = r'IMG_0846_JPG_jpg.rf.ce1204d0cc2fd76a2a30a428bb66c2af.jpg'
# # img_name1 = r'IMG_0847_JPG_jpg.rf.f0d342f21a8573ccd2c39b24e258cafe.jpg'
# # img_name2 = r'IMG_0848_JPG_jpg.rf.d6d13a202b50378762756889363d2fef.jpg'
# # img_name2 = r'IMG_0845_JPG_jpg.rf.5bf18cf1effa5abce28e1c4d93dc7dff.jpg'
# '''
# SSIM: 0.549671291143662
# Histogram Similarity: 0.8985382908603161
# pHash similarity (distance): 8
# Detected shift (x, y): ((-12.94651216700197, -30.21279684981579), 0.7554708497226238)
# '''
# # img_name1 = r'IMG_2460_JPG_jpg.rf.362f7ea90ed98293b6726dca9c9fa22d.jpg'
# # img_name2 = r'IMG_2461_JPG_jpg.rf.d5891a9e05d20033bb021569b88362fc.jpg'
# '''
# SSIM: 0.7514438351005026
# Histogram Similarity: 0.7662142931378835
# pHash similarity (distance): 26
# '''
# # img_name1 = r'IMG_20200118_100636_jpg.rf.c07853f314a2b33614884b7b65ce15ef.jpg'
# # img_name2 = r'IMG_20200118_100640_jpg.rf.6a955292322455a90a7ccbe35641b63c.jpg'
#
#
# img_name1 = r'IMG_0767_JPG_jpg.rf.95eb86b8b75d48d31efea07fc7d209a3.jpg'
# img_name2 = r'IMG_0768_JPG_jpg.rf.3a9b53b19633747e7f3f088029d31c3f.jpg'
# img_path1 = os.path.join(img_dir, img_name1)
# img_path2 = os.path.join(img_dir, img_name2)
#
# # 读取两张图片（用 OpenCV 读取以进行直方图计算）
# imageA = cv2.imread(img_path1)
# imageB = cv2.imread(img_path2)

# # 计算SSIM
# similarity = calculate_ssim(imageA, imageB)
# print(f"SSIM: {similarity}")
#
# # 计算直方图相似度
# hist_similarity = calculate_histogram_similarity(imageA, imageB)
# print(f"Histogram Similarity: {hist_similarity}")
#
# # 计算 pHash 相似度
# phash_similarity = calculate_phash_similarity(img_path1, img_path2)
# print(f"pHash similarity (distance): {phash_similarity}")


# # 找到两张图片之间的平移
# translation = find_translation(imageA, imageB)
# if translation is not None:
#     print(f"Translation vector: {translation}")
# else:
#     print("No translation found or insufficient matches.")

# shift = phase_correlation(imageA, imageB)
# print(f"Detected shift (x, y): {shift}")
# import timm
# # # print(timm.list_models())
# # model = timm.create_model('convnext_tiny', pretrained=True)
# # print(model)
# for m in timm.list_models():
#     print(m)
#
# model = timm.create_model('swin_large_patch4_window12_384', pretrained=True)

from timm.models import create_model
import utils

model = create_model('repvit_m0_9')
utils.replace_batchnorm(model)

