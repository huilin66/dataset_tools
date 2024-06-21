import os, cv2
import exifread
import requests
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from tqdm import tqdm


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
    closest_point = gdf.loc[gdf['distance'].idxmin()]

    name = closest_point['NAME_TC']
    distance = closest_point['distance']
    search_url = closest_point['NSEARCH061']

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

def is_image_in_image(small_image_path, large_image_path):
    # Read the images from the given file paths
    small_image = cv2.imread(small_image_path)
    large_image = cv2.imread(large_image_path)

    # Check if images are read correctly
    if small_image is None:
        print(f"Error: Could not read the small image from {small_image_path}")
        return False
    if large_image is None:
        print(f"Error: Could not read the large image from {large_image_path}")
        return False

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


if __name__ == '__main__':
    pass
    img_dir = r'E:\data\0417_signboard\data0504'
    dst_dir = r'E:\data\0417_signboard\data0504_match'
    shp_path = r'E:\data\0417_signboard\database\billboard.shp'

    # imgs_match(img_dir, dst_dir, shp_path)

    gps_info = get_gps_info(os.path.join(img_dir, 'MVIMG_20240504_142044.jpg'))
    print(gps_info)