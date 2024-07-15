import os, cv2, io
import exifread
import matplotlib.pyplot as plt
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from tqdm import tqdm
import fitz
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
import time
# 中文名：NAME_TC,
# 招牌类型：NSEARCH011,
# 长：NSEARCH031
# 宽：NSEARCH041
# 厚：NSEARCH051
# 照片：NSEARCH061
# 经度：LONGITUDE
# 纬度：LATITUDE
# 坐标北：NORTHING
# 坐标东：EASTING
# wall signboard, projecting signboard


def search(shp_path, gps_info):
    # point = Point(gps_info['latitude'], gps_info['longitude'])


    gdf = gpd.read_file(shp_path)

    shp_crs = gdf.crs
    # print(f"Shapefile CRS: {shp_crs}")

    # Define a transformer to convert from WGS84 (lat/lon) to the Shapefile's CRS
    transformer = Transformer.from_crs("epsg:4326", shp_crs, always_xy=True)

    # Transform the GPS point to the Shapefile's CRS
    transformed_point = transformer.transform( gps_info['longitude'], gps_info['latitude'])
    # print(f"Transformed Point: {transformed_point}")

    # Create a point geometry for the transformed GPS location
    point = Point(transformed_point)


    gdf['distance'] = gdf.geometry.distance(point)
    # closest_point = gdf.loc[gdf['distance'].idxmin()]

    closest_points = gdf[gdf['distance'] < 10]

    name = closest_points['NAME_TC'].to_list()
    distance = closest_points['distance'].to_list()
    search_url = closest_points['NSEARCH061'].to_list()

    return distance, search_url


def get_gps_info(img_path):
    def convert_ratio_to_float(ratio):
        """Convert a Ratio object to a float."""
        return float(ratio.num) / float(ratio.den)
    def convert_dms_to_decimal(degrees, minutes, seconds):
        """Convert degrees, minutes, and seconds to a decimal degree."""
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
    """Extract and convert GPS info from EXIF tags."""
    with open(img_path, 'rb') as f:
        exif_tags = exifread.process_file(f)

    gps_info = {}
    if 'GPS GPSLatitude' in exif_tags and 'GPS GPSLongitude' in exif_tags:
        # Extract latitude
        lat_ratios = exif_tags['GPS GPSLatitude'].values
        lat_degrees = convert_ratio_to_float(lat_ratios[0])
        lat_minutes = convert_ratio_to_float(lat_ratios[1])
        lat_seconds = convert_ratio_to_float(lat_ratios[2])
        latitude = convert_dms_to_decimal(lat_degrees, lat_minutes, lat_seconds)

        # Check latitude reference
        lat_ref = exif_tags['GPS GPSLatitudeRef'].printable
        if lat_ref != 'N':
            latitude = -latitude

        # Extract longitude
        lon_ratios = exif_tags['GPS GPSLongitude'].values
        lon_degrees = convert_ratio_to_float(lon_ratios[0])
        lon_minutes = convert_ratio_to_float(lon_ratios[1])
        lon_seconds = convert_ratio_to_float(lon_ratios[2])
        longitude = convert_dms_to_decimal(lon_degrees, lon_minutes, lon_seconds)

        # Check longitude reference
        lon_ref = exif_tags['GPS GPSLongitudeRef'].printable
        if lon_ref != 'E':
            longitude = -longitude

        gps_info['latitude'] = latitude
        gps_info['longitude'] = longitude
    # print(gps_info)
    return gps_info


def download_pdf(url, filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the file in write-binary mode and write the response content to the file
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded and saved as {filename}")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def extract_images_from_pdf(pdf_path):
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    image_arrays = []

    for page_number in range(len(pdf_document)):
        # 获取页面
        page = pdf_document.load_page(page_number)
        # 获取页面上的所有图像
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            # 提取图像的索引号和图片信息
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 将图片数据读入 PIL 图像
            image = Image.open(io.BytesIO(image_bytes))
            # 将 PIL 图像转换为 NumPy 数组
            image_array = np.array(image)
            image_arrays.append(image_array)
            # plt.imshow(image_array)
            # plt.show()
    return image_arrays[0]

def is_image_in_image(small_image, large_image):
    # Convert images to grayscale
    small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    large_image_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

    # Use matchTemplate to find the small image in the large image
    result = cv2.matchTemplate(large_image_gray, small_image_gray, cv2.TM_CCOEFF_NORMED)

    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Define a threshold for matching
    threshold = 0.8

    # Check if the best match exceeds the threshold
    if max_val >= threshold:
        print(f"Image found with max correlation value: {max_val}")
        return True
    else:
        print(f"Image not found. Max correlation value: {max_val}")
        return False

def imgs_match(img_dir, dst_dir, shp_path):
    os.makedirs(dst_dir, exist_ok=True)
    imgs_list = [img_name for img_name in os.listdir(img_dir) if img_name.endswith('.jpg')]
    for img_name in imgs_list:
        img_path = os.path.join(img_dir, img_name)

        gps_info = get_gps_info(img_path)
        if len(gps_info) == 0:
            continue
        distance, dst_url = search(shp_path, gps_info)

        dst_path = os.path.join(dst_dir, img_name.replace('.jpg', '_dis%.2f.pdf'%distance))
        download_pdf(dst_url, dst_path)
        large_image = extract_images_from_pdf(dst_path)

        small_image = np.array(Image.open(img_path))
        result = is_image_in_image(small_image, large_image)
        print(result)

def ladybug2gps(img_path, time_csv_path, idx, info_csv_path):
    def dms_to_decimal(dms):
        degrees, minutes, seconds1, seconds2 = dms.split('.')
        degrees = int(degrees)
        minutes = int(minutes)
        seconds = float(seconds1+'.'+seconds2)
        return degrees + minutes / 60 + seconds / 3600
    df_time = pd.read_csv(time_csv_path, header=0, index_col=None)
    row = df_time.loc[idx*2]
    time_lady = row.loc[' CAMERA TIME'].replace(' ', '')
    df_info = pd.read_csv(info_csv_path, header=[0, 1], index_col=None)
    print(time_lady)
    time_lady = datetime.strptime(time_lady, '%H:%M:%S.%f')


    df_info['time'] = df_info['UTCTime'].apply(lambda x: pd.to_datetime(x, unit='s').dt.strftime('%H:%M:%S.%f'))
    df_info['time_diff'] = df_info['time'].apply(lambda x: abs(datetime.strptime(x, '%H:%M:%S.%f') - time_lady))


    closest_row = df_info.loc[df_info['time_diff'].idxmin()]
    print(closest_row['time_diff'])

    Latitude = closest_row.loc['Latitude'][0]
    Longitude = closest_row.loc['Longitude'][0]
    Latitude = dms_to_decimal(Latitude)
    Longitude = dms_to_decimal(Longitude)

    gps_info = {}
    gps_info['latitude'] = Latitude
    gps_info['longitude'] = Longitude
    return gps_info

if __name__ == '__main__':
    pass
    # img_dir = r'E:\data\0417_signboard\data0504'
    # dst_dir = r'E:\data\0417_signboard\data0504_match'
    # shp_path = r'E:\data\0417_signboard\database\billboard.shp'


    # img_path = r'E:\data\0417_signboard\1719366000603.jpg'
    # gps_info = get_gps_info(img_path)
    # print(gps_info)


    # imgs_match(img_dir, dst_dir, shp_path)

    # gps_info = get_gps_info(os.path.join(img_dir, 'MVIMG_20240504_142044.jpg'))
    # print(gps_info)

    # data_path = r'E:\data\0417_signboard\VMMS\ladybug\data.csv'
    # df = pd.read_csv(data_path, sep='')
    # print(df.info)

    shp_path = r'E:\data\0417_signboard\database\billboard.shp'
    info_csv_path = r'E:\data\0417_signboard\VMMS\ladybug\data.csv'

    #
    # # img_path = r'E:\data\0417_signboard\VMMS\ladybug\4\panoramic\ladybug_21505973_20240621_165137_Panoramic_000010_3019_103-3345.jpg'
    # # time_csv_path = r'E:\data\0417_signboard\VMMS\ladybug\ladybug_frame_gps_info_16.txt'
    # # dst_dir = r'E:\data\0417_signboard\VMMS\ladybug\4\dst'
    #
    # # img_path = r'E:\data\0417_signboard\VMMS\ladybug\3\panoramic\ladybug_21505973_20240621_164818_Panoramic_000005_1141_039-5080.jpg'
    # # time_csv_path = r'E:\data\0417_signboard\VMMS\ladybug\ladybug_frame_gps_info_124.txt'
    # # dst_dir = r'E:\data\0417_signboard\VMMS\ladybug\3\dst'
    #
    img_path = r'E:\data\0417_signboard\VMMS\ladybug\4\panoramic\ladybug_21505973_20240621_165137_Panoramic_000015_3139_107-3345.jpg'
    time_csv_path = r'E:\data\0417_signboard\VMMS\ladybug\ladybug_frame_gps_info_16.txt'
    dst_dir = r'E:\data\0417_signboard\VMMS\ladybug\4\dst'
    idx = 15

    img_name = os.path.basename(img_path)
    os.makedirs(dst_dir, exist_ok=True)


    gps_info = ladybug2gps(img_path, time_csv_path, idx, info_csv_path)
    print(gps_info)
    distances, dst_urls = search(shp_path, gps_info)

    for distance, dst_url in zip(distances, dst_urls):
        dst_path = os.path.join(dst_dir, img_name.replace('.jpg', '_dis%.2f.pdf' % distance))
        download_pdf(dst_url, dst_path)
        large_image = extract_images_from_pdf(dst_path)

        small_image = np.array(Image.open(img_path))
        result = is_image_in_image(small_image, large_image)
        print(result)