import os
import shutil
import pandas as pd
from tqdm import tqdm
from YOLO_slicing.slicing import slice_image

def check_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def yolo_slice(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir, tp_dir,
               slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5):
    check_dir(output_img_dir)
    check_dir(output_txt_dir)
    check_dir(tp_dir)
    img_list = os.listdir(input_img_dir)
    suffix = '.' + img_list[0].split('.')[-1]
    for img_name in tqdm(img_list):
        txt_name = img_name.replace(suffix, '.txt')
        input_image_path = os.path.join(input_img_dir, img_name)
        input_txt_path = os.path.join(input_txt_dir, txt_name)
        # tp_image_path = os.path.join(tp_dir, img_name)
        # tp_txt_path = os.path.join(tp_dir, txt_name)
        # output_image_path = os.path.join(output_img_dir, img_name)
        # output_txt_path = os.path.join(output_txt_dir, txt_name)

        slice_image(input_image_path, input_txt_path, tp_dir, slice_w, slice_h, overlap_w, overlap_h)

    img_list = [file_name for file_name in os.listdir(tp_dir) if file_name.endswith(suffix)]
    for img_name in tqdm(img_list):
        txt_name = img_name.replace(suffix, '.txt')
        tp_image_path = os.path.join(tp_dir, img_name)
        tp_txt_path = os.path.join(tp_dir, txt_name)
        output_image_path = os.path.join(output_img_dir, img_name)
        output_txt_path = os.path.join(output_txt_dir, txt_name)

        shutil.move(tp_image_path, output_image_path)
        shutil.move(tp_txt_path, output_txt_path)


def get_dataset(input_csv, output_csv):
    def replace_filename(row):
        idx = row.name % 3
        base_filename = (row['filename'].replace('.jpg', '_%d_%d_%d_%d.jpg'%(idx*960, 0, (idx+2)*960, 1920)).
                         replace(os.path.dirname(input_csv), os.path.dirname(output_csv)))
        return base_filename
    df = pd.read_csv(input_csv, header=None, index_col=None, names=['filename'])

    new_df = pd.concat([df] * 3, ignore_index=True)
    new_df['filename'] = new_df.apply(replace_filename, axis=1)

    new_df.to_csv(output_csv, header=None, index=None)


if __name__ == '__main__':
    pass

    # yolo_slice(input_img_dir=r'E:\data\0318_fireservice\data0327\images',
    #            input_txt_dir=r'E:\data\0318_fireservice\data0327\labels',
    #            output_img_dir=r'E:\data\0318_fireservice\data0327\images_slice',
    #            output_txt_dir=r'E:\data\0318_fireservice\data0327\labels_slice',
    #            tp_dir=r'E:\data\0318_fireservice\data0327\tp',
    #            slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5)

    get_dataset(input_csv=r'E:\data\0318_fireservice\data0327\train.txt',
                output_csv=r'E:\data\0318_fireservice\data0327slice\train.txt',)
    get_dataset(input_csv=r'E:\data\0318_fireservice\data0327\val.txt',
                output_csv=r'E:\data\0318_fireservice\data0327slice\val.txt',)