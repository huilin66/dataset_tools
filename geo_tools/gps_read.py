import exifread
import re


def read():
    GPS = {}
    date = ''
    f = open(r"E:\data\0417_signboard\data0507\src\FLIR0764.jpg", 'rb')
    contents = exifread.process_file(f)
    for key in contents:
        if key == "GPS GPSLongitude":
            print("经度 =", contents[key], contents['GPS GPSLatitudeRef'])
        elif key == "GPS GPSLatitude":
            print("纬度 =", contents[key], contents['GPS GPSLongitudeRef'])
        else:
            print(key)

if __name__ == '__main__':
    read()