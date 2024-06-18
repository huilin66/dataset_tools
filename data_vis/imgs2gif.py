import os
import imageio
from PIL import Image, ImageDraw, ImageFont

# 文件夹路径
img_dir = r'E:\data\0319_demodata\BD-Detection\images'
save_path = r'E:\data\0319_demodata\BD-Detection\images.gif'
img_list = ['image_src.jpg', 'map_src.jpg', 'map_dense.jpg']
text_list = ['原始全景图片', '原始映射结果', '插值后映射结果']

# 文字字体和大小
font_path = r"C:\\Windows\\Fonts\\times.ttf"
font_size = 20
font = ImageFont.truetype(font_path, font_size)
# 文字颜色
text_color = (255, 0, 0)



# 读取图片并在每张图片上添加文字
images = []
for idx, img_name in enumerate(img_list):
    img_path = os.path.join(img_dir, img_name)
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    # 计算文字位置（居中）
    text_width, text_height = draw.textsize(text_list[idx], font=font)
    position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
    # 在图片上添加文字
    draw.text(position, text_list[idx], fill=text_color, font=font)
    images.append(img)

# 创建GIF
imageio.mimsave(save_path, images, 'GIF', duration=1000)

# print("GIF制作完成：", gif_file)

# # 获取文件夹内的所有图片文件
# image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
#
# # 图片文件排序（按文件名排序）
# image_files.sort()

# print(image_files)

# # GIF文件路径
# gif_file = 'output.gif'
#
# # 文字内容
# text = "Sample Text"
# # 文字字体和大小
# font_path = "path/to/your/font.ttf"
# font_size = 20
# font = ImageFont.truetype(font_path, font_size)
# # 文字颜色
# text_color = (255, 255, 255)
#
# # 读取图片并在每张图片上添加文字
# images = []
# for image_file in image_files:
#     img = Image.open(image_file)
#     draw = ImageDraw.Draw(img)
#     # 计算文字位置（居中）
#     text_width, text_height = draw.textsize(text, font=font)
#     position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
#     # 在图片上添加文字
#     draw.text(position, text, fill=text_color, font=font)
#     images.append(img)
#
# # 创建GIF
# imageio.mimsave(gif_file, images)
#
# print("GIF制作完成：", gif_file)
