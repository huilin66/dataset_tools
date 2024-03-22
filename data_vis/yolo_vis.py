import os
import numpy as np
import cv2
from tqdm import tqdm

cats = {
    0: 'background',
    1: 'crack',
    2: 'water_seepage',
    3: 'finishes_peeling',
    4: 'concrete_spalling',
}

# cats = {
#     0: 'background',
#     1: 'crack',
#     2: 'mold',
#     3: 'peeling_paint',
#     4: 'stairstep_crack',
#     5: 'water_seepage',
#     6: 'spall',
# }

# 修改输入图片文件夹
img_folder = r"E:\Huilin\2308_concretespalling\data\merge_data\yolo_fomat_c4\images\val"
img_list = os.listdir(img_folder)
img_list.sort()

# 修改输入标签文件夹
label_folder = r"E:\Huilin\2308_concretespalling\data\merge_data\yolo_fomat_c4\labels\val"
label_list = os.listdir(label_folder)
label_list.sort()

# 输出图片文件夹位置
path = os.getcwd()
output_folder = r"E:\Huilin\2308_concretespalling\data\merge_data\yolo_fomat_c4\images\val_vis"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

colormap = [(0, 255, 0), (132, 112, 255), (0, 191, 255)]  # 色盘，可根据类别添加新颜色

# 坐标转换
def xywh2xyxy(x, w1, h1, img):
    label, x, y, w, h = x
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), colormap[1], 2)
    cv2.putText(img, cats[int(label)], (int(top_left_x), int(top_left_y) +10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img

if __name__ == '__main__':
    for i in tqdm(range(len(img_list))):
        image_path = os.path.join(img_folder, img_list[i])
        label_path = os.path.join(label_folder, label_list[i])
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        with open(label_path, 'r') as f:
            lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            for x in lb:
                img = xywh2xyxy(x, w, h, img)
            save_path = image_path.replace(img_folder, output_folder)
            cv2.imwrite(save_path, img)
