import os
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
def img_read(img_path):
    img = io.imread(img_path)
    return img

def dir_shape_sta(imgs_path, save_path):
    '''
    统计影像shape分布，并绘制二维分布图，以文件夹方式进行
    :param imgs_path: 影像文件夹路径
    :param save_path: csv结果路径
    :return:
    '''
    print('images shape sta:')
    imgs_list = [os.path.join(imgs_path, file_name) for file_name in os.listdir(imgs_path)]
    img_shape_df = list_shape_sta(imgs_list, save_path)
    print('finish\n')
    return img_shape_df


def list_shape_sta(imgs_list, save_path):
    '''
    统计影像shape分布，并绘制二维分布图，以list方式进行
    :param imgs_list: 影像文件路径list
    :param save_path: csv结果路径
    :return:
    '''
    new_img_list = []
    shape_df = pd.DataFrame(None, columns=['img_height', 'img_width'])
    with tqdm(imgs_list) as pbar:
        pbar.set_description('shape sta ')
        for index, img_path in enumerate(pbar):
            try:
                img = img_read(img_path)
            except Exception as e:
                print(img_path, e)
                continue
            shape_df.loc[index] = [img.shape[0], img.shape[1]]
            new_img_list.append(Path(img_path).stem)

    sns.jointplot(x='img_height', y='img_width', data=shape_df)
    plt.savefig(save_path)

    shape_df['image'] = [os.path.basename(gt_path) for gt_path in new_img_list]
    shape_df.to_csv(save_path.replace('.png', '.csv'))
    print('save to %s' % save_path)
    return shape_df


if __name__ == '__main__':
    pass
    # dir_shape_sta(r'E:\data\tp\car_det_train\car_det_train\input_path',
    #               r'E:\data\tp\car_det_train\car_det_train\input_path.csv')
    # dir_shape_sta(r'E:\data\tp\multi_modal_airplane_train\rgb',
    #               r'E:\data\tp\multi_modal_airplane_train\rgb.csv')
    # dir_shape_sta(r'E:\data\tp\multi_modal_airplane_train\sar',
    #               r'E:\data\tp\multi_modal_airplane_train\sar.csv')
    dir_shape_sta(r'E:\data\tp\multi_modal_airplane_train\img',
                  r'E:\data\tp\multi_modal_airplane_train\img.csv')