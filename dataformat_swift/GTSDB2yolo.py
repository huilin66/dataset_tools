import os
import pandas as pd
from skimage import io
import imageio
from tqdm import tqdm

def convert_xyxy2yolo(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = (box[2] - box[0])
    h = (box[3] - box[1])

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

def get_yolo_box(file_path, box):
    img = imageio.imread(file_path)
    yolo_box = convert_xyxy2yolo([img.shape[1], img.shape[0]], box)
    return yolo_box

def get_yolo(img_dir, gt_path, gt_output_dir, img_output_dir):
    pass
    if not os.path.exists(gt_output_dir):
        os.makedirs(gt_output_dir)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    df_gt = pd.read_csv(gt_path, sep=';', header=None, index_col=None,
                        names=['file_name', 'x1', 'y1', 'x2', 'y2', 'cat_id'])

    box_dict = {}
    for idx, row in tqdm(df_gt.iterrows()):
        file_name = row['file_name']
        yolo_box = get_yolo_box(os.path.join(img_dir, file_name),
                           [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])])
        if file_name not in box_dict.keys():
            box_dict[file_name] = []
        box_dict[file_name].append(yolo_box+[int(row['cat_id'])])

    for k,vs in tqdm(box_dict.items()):
        file_name = k
        df_gt_s = pd.DataFrame(None, columns=['x1', 'y1', 'x2', 'y2', 'cat_id'])
        for v in vs:
            df_gt_s.loc[len(df_gt_s)] = v
        df_gt_s.to_csv(os.path.join(gt_output_dir, file_name.replace('.ppm', '.txt')))
        img = imageio.imread(os.path.join(img_dir, file_name))
        io.imsave(os.path.join(img_output_dir, file_name.replace('.ppm', '.png')), img)


if __name__ == '__main__':
    pass
    root_dir = r'E:\data\0416_trafficsign\GTSDB'
    img_dir = os.path.join(root_dir, 'imgs')
    gt_path = os.path.join(root_dir, 'gt.txt')
    image_dir = os.path.join(root_dir, 'images')
    gt_output_dir = os.path.join(root_dir, 'labels')
    get_yolo(img_dir, gt_path, gt_output_dir, image_dir)
