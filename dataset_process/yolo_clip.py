import shutil
from pathlib import Path

import yaml
from PIL import Image
import os.path

from tqdm import tqdm

rootdir = r'D:\Data\i1'  # 原始图片文件夹
savedir = r'D:\Data\i2'  # 保存图片文件夹
ConfigPath = "../config.yaml"  # 配置文件(记录图片尺寸)
dis = 1280
leap = 1280

def yolo_clip(input_dir, output_dir,):
    # 创建输出文件夹
    if Path(savedir).exists():
        shutil.rmtree(savedir)
    os.mkdir(savedir)

    for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
        filenames.sort()
        for filename in tqdm(filenames):
            currentPath = os.path.join(parent, filename)
            suffix = currentPath.split('.')[-1]
            if suffix == 'jpg' or suffix == 'png' or suffix == 'JPG' or suffix == 'PNG':
                img = Image.open(currentPath)
                width = img.size[0]
                height = img.size[1]
                i = j = 0
                for i in range(0, width, leap):
                    for j in range(0, height, leap):
                        box = (i, j, i+dis, j+dis)
                        image = img.crop(box)  # 图像裁剪
                        image.save(savedir + '/' + filename.split(".")[0] + "__" + str(i) + "__" + str(j) + ".jpg")

    # 将图片长宽写入配置文件
    pic_context = {
                    'width': width,
                    'height': height
                    }
    with open(ConfigPath, "w", encoding="utf-8") as f:
        yaml.dump(pic_context, f)



if __name__ == '__main__':

    pass