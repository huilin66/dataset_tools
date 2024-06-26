import os
import numpy as np
import pandas as pd


# class_dict = {
#     0 : 'speed limit 20 (prohibitory)',
#     1 : 'speed limit 30 (prohibitory)',
#     2 : 'speed limit 50 (prohibitory)',
#     3 : 'speed limit 60 (prohibitory)',
#     4 : 'speed limit 70 (prohibitory)',
#     5 : 'speed limit 80 (prohibitory)',
#     6 : 'restriction ends 80 (other)',
#     7 : 'speed limit 100 (prohibitory)',
#     8 : 'speed limit 120 (prohibitory)',
#     9 : 'no overtaking (prohibitory)',
#     10 : 'no overtaking (trucks) (prohibitory)',
#     11 : 'priority at next intersection (danger)',
#     12 : 'priority road (other)',
#     13 : 'give way (other)',
#     14 : 'stop (other)',
#     15 : 'no traffic both ways (prohibitory)',
#     16 : 'no trucks (prohibitory)',
#     17 : 'no entry (other)',
#     18 : 'danger (danger)',
#     19 : 'bend left (danger)',
#     20 : 'bend right (danger)',
#     21 : 'bend (danger)',
#     22 : 'uneven road (danger)',
#     23 : 'slippery road (danger)',
#     24 : 'road narrows (danger)',
#     25 : 'construction (danger)',
#     26 : 'traffic signal (danger)',
#     27 : 'pedestrian crossing (danger)',
#     28 : 'school crossing (danger)',
#     29 : 'cycles crossing (danger)',
#     30 : 'snow (danger)',
#     31 : 'animals (danger)',
#     32 : 'restriction ends (other)',
#     33 : 'go right (mandatory)',
#     34 : 'go left (mandatory)',
#     35 : 'go straight (mandatory)',
#     36 : 'go right or straight (mandatory)',
#     37 : 'go left or straight (mandatory)',
#     38 : 'keep right (mandatory)',
#     39 : 'keep left (mandatory)',
#     40 : 'roundabout (mandatory)',
#     41 : 'restriction ends (overtaking) (other)',
#     42 : 'restriction ends (overtaking (trucks)) (other)',
# }

class_dict = {
    0 : 'surface',
    1 : 'frame',
}
def random_select(img_dir, dst_dir, train_ratio=0.9, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]
        # file_list = [os.path.join(os.path.basename(img_dir), filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_train.to_csv(os.path.join(dst_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(dst_dir, 'val.txt'), header=None, index=None)

def gap_select(img_dir, dst_dir, train_ratio=0.9, full_path=True):
    file_list = os.listdir(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]

    interval = len(file_list) // int((1-train_ratio)*100)  # 计算间隔

    val_list = file_list[::interval]  # 使用切片功能取出等间隔的元素
    train_list = [item for i, item in enumerate(file_list) if i % interval != 0]  # 将其余元素保存到train_list

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_train.to_csv(os.path.join(dst_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(dst_dir, 'val.txt'), header=None, index=None)

def get_class(class_dict, class_path):
    df_train = pd.DataFrame({'cat_name': list(class_dict.values())})
    df_train.to_csv(class_path, header=None, index=None)

if __name__ == '__main__':
    pass
    # random_select(r'E:\data\0318_fireservice\data0318\images',
    #               r'E:\data\0318_fireservice\data0318')
    # gap_select(r'E:\data\0318_fireservice\data0318\images',
    #               r'E:\data\0318_fireservice\data0318')

    # random_select(r'E:\data\0416_trafficsign\GTSDB\images',
    #               r'E:\data\0416_trafficsign\GTSDB')
    # get_class(class_dict, r'E:\data\0416_trafficsign\GTSDB\classes.txt')


    # random_select(r'E:\data\0417_signboard\data0521_m\yolo\images',
    #               r'E:\data\0417_signboard\data0521_m\yolo')

    # random_select(r'E:\data\1123_thermal\thermal data\datasets\moisture\images',
    #               r'E:\data\1123_thermal\thermal data\datasets\moisture')


    random_select(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\images',
                  r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection4')