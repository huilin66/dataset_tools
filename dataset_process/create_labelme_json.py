import json
import base64
import os.path

from PIL import Image

def create_json(img_path, json_path):
    # 读取图像
    img = Image.open(img_path)

    # 初始化json字典
    cur_json_dict = {
        "version": "5.3.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageHeight": img.size[1],
        "imageWidth": img.size[0]
    }

    # 将图像转换为base64编码
    with open(img_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    cur_json_dict["imageData"] = image_data

    # 将字典保存为json文件
    with open(json_path, 'w') as json_file:
        json.dump(cur_json_dict, json_file)

if __name__ == '__main__':
    pass
    create_json(r'E:\data\0111_testdata\data\img_w\V1_6F_DJI_0957_W.JPG',
                r'E:\data\0111_testdata\data\img_w\V1_6F_DJI_0957_W_cp.json',
                )