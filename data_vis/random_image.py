from PIL import Image, ImageDraw

# 创建灰色背景图像
width, height = 1080, 720
background_color = (128, 128, 128)  # 灰色 (R, G, B)
image = Image.new("RGB", (width, height), background_color)
draw = ImageDraw.Draw(image)

# 定义正方形的尺寸和空隙
square_size = 200
gap = 20  # 方块之间的空隙
square_color = (255, 255, 255)  # 白色 (R, G, B)

# 计算每行的方块数量和行间距
num_squares = 6
squares_per_row = num_squares // 2  # 每行 3 个方块
total_squares_width = squares_per_row * square_size  # 每行所有正方形的总宽度
total_gaps_width = (squares_per_row - 1) * gap  # 每行所有空隙的总宽度
remaining_space = width - (total_squares_width + total_gaps_width)  # 每行剩余空间
spacing_x = remaining_space // 2  # 每行左右两侧的边距
spacing_y = (height - 2 * square_size - gap) // 2  # 垂直方向的总空隙

# 绘制 6 个正方形，分为两行
for row in range(2):  # 两行
    for col in range(squares_per_row):  # 每行 3 个方块
        # 计算每个正方形的左上角坐标
        x1 = spacing_x + col * (square_size + gap)
        y1 = spacing_y + row * (square_size + gap)
        x2 = x1 + square_size
        y2 = y1 + square_size
        # 绘制正方形
        draw.rectangle([x1, y1, x2, y2], fill=square_color)

# 保存图像
image.save("output_image_two_rows.png")

# # 显示图像
# image.show()