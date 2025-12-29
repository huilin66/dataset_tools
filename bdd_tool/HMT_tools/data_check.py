from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread

def get_exif_with_gps(path, show=True):
    img = Image.open(path)
    exif_data = img._getexif()
    if not exif_data:
        return None

    exif = {}
    gps = {}

    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == "GPSInfo":
            for gps_tag in value:
                sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                gps[sub_tag] = value[gps_tag]
        else:
            exif[tag_name] = value
    if show:
        print("GPS Info:")
        for k, v in gps.items():
            print(f"{k}: {v}")
    return exif, gps

def check_pose(path):
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=True)

    keys = [k for k in tags.keys() if any(s in k.lower() for s in ["yaw","pitch","roll","gimbal","flight","drone","xmp"])]
    for k,v in tags.items():
        print(f"{k}: {v}")
    print("\n".join(keys[:200]))

def check_pose2(img_path):
    b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
    a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"

    aa=["\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"]
    bb=["\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"]

    #xml format to save EXIF的数据规范
    # aa ['<rdf:Description ']
    print("aa",aa)
    # bb ['</rdf:Description>']
    print("bb",bb)



    # rb是读取二进制文件
    img = open(img_path, 'rb')
    # bytearray() 方法返回一个新字节数组
    data = bytearray()
    #标识符,
    flag = False

    for i in img.readlines():
        # 按行读取二进制信息，标签成对出现
        if a in i:
            flag = True
        if flag:
            #把第i行数据复制到新数组中
            data += i
        if b in i:
            break
    print("data",data)

    if len(data) > 0:
        data = str(data.decode('ascii'))
        print(data)
        #filter()函数用于过滤序列，过滤掉不符合条件的元素，返回符合条件的元素组成新列表。
        #filter(function,iterable) ,function -- 判断函数。iterable -- 可迭代对象
        #python允许用lambda关键字创造匿名函数。
        # 在 lambda 关键字之后、冒号左边为参数列表，可不带参数，也可有多个参数。若有多个参数，则参数间用逗号隔开，冒号右边为 lambda 表达式的返回值。
        #left--->right
        # judge condition 'drone-dji:' in x
        lines = list(filter(lambda x: 'drone-dji:' in x, data.split("\n")))
        print("lines",lines)
        dj_data_dict = {}
        for d in lines:
            # remove 'drone-dji:'
            d = d.strip()[10:]
            # k is name
            # v is value
            k, v = d.split("=")
            print(f"{k} : {v}")
            dj_data_dict[k] = v



# -*- coding: utf-8 -*-
"""
@Time ： 2023/06/21 11:52
@Auth ： RS迷途小书童
@File ：Read Image.py
@IDE ：PyCharm
@Purpose：读取图片信息
@Web：博客地址:https://blog.csdn.net/m0_56729804
"""
# import exifread
# from osgeo import gdal



def Get_Image_Yaw_angle(file_path):
    """
    :param file_path: 输入图片路径
    :return: 图片的偏航角
    """
    # 获取图片偏航角
    print("----------------------------------大疆exifread信息---------------------------------")
    # 定义字节模式 b 和 a，用于查找大疆EXIF数据的起始和结束标记
    b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
    a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"
    # 打开图片文件，以二进制模式读取
    img = open(file_path, 'rb')
    # 初始化一个字节数组用于存储EXIF数据
    data = bytearray()
    # 初始化一个标志，用于判断是否已经找到EXIF数据的起始标记
    flag = False
    # 逐行读取图片文件内容
    for line in img.readlines():
        # 如果当前行包含EXIF数据的起始标记，则设置标志为True
        if a in line:
            flag = True
            # 如果标志为True，则将当前行添加到EXIF数据中
        if flag:
            data += line
            # 如果当前行包含EXIF数据的结束标记，则跳出循环
        if b in line:
            break
            # 如果提取到的EXIF数据不为空
    dj_data_dict = {}
    # 遍历过滤后的行，并提取键值对存入字典中
    if len(data) > 0:
        # 将字节数据解码为ASCII字符串
        data = str(data.decode('ascii'))
        # 过滤出包含drone-dji的行，并分割每行为键值对
        lines = list(filter(lambda x: 'drone-dji:' in x, data.split("\n")))
        # 初始化一个空字典用于存储提取到的数据
        for d in lines:
            d = d.strip()[10:]  # 去除每行的前后空格和'\n'字符，并从第10个字符开始处理（因为drone-dji:占据了前9个字符）
            k, v = d.split("=")  # 将当前行分割为键和值两部分
            print(f"{k} : {v}")  # 打印键和值
            dj_data_dict[k] = v  # 将键值对存入字典中
    return dj_data_dict.get('YawAngle')  # 返回偏航角的值。如果未找到偏航角，则返回None。

if __name__ == '__main__':
    pass
    image_path = r"E:\data\thesis\HTM\collected data\DJI_202512161540_008_filter\DJI_20251216155812_0537_V.JPG"
    # params = extract_dji_pose_from_xmp(image_path)
    # check_pose(image_path)
    check_pose2(image_path)

    # import subprocess, shlex
    # cmd = r'exiftool -G -s DJI_20251216154233_0006_V.JPG'
    # out = subprocess.check_output(shlex.split(cmd), text=True, errors="ignore")
    # for line in out.splitlines():
    #     if any(k in line.lower() for k in ["yaw","pitch","roll","gimbal","flight","heading","imgdirection"]):
    #         print(line)