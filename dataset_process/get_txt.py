import os
import pandas as pd

def dir2txt(input_dir, txt_path):
    img_list = os.listdir(input_dir)
    img_list = [os.path.join(input_dir, file_name) for file_name in img_list]
    df_train = pd.DataFrame({'filename': img_list})
    df_train.to_csv(txt_path, header=None, index=None)

def cats2txt(cats_list, txt_path):
    df_train = pd.DataFrame({'filename': cats_list})
    df_train.to_csv(txt_path, header=None, index=None)

if __name__ == '__main__':
    pass
    # img_dir = r'E:\data\0417_signboard\data0417\yolo\images\train'
    # txt_path = r'E:\data\0417_signboard\data0417\yolo\train.txt'
    # dir2txt(img_dir, txt_path)
    #
    # img_dir = r'E:\data\0417_signboard\data0417\yolo\images\val'
    # txt_path = r'E:\data\0417_signboard\data0417\yolo\val.txt'
    # dir2txt(img_dir, txt_path)

    cats_list = ['boardin', 'boardout']
    txt_path = r'E:\data\0417_signboard\data0417\yolo\classes.txt'
    cats2txt(cats_list, txt_path)