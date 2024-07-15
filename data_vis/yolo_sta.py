import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


shp_rate_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.4, 2.6, 3, 3.5, 4, 5]

# center_x center_y width height
def yolo_sta(gt_dir, result_dir, ref_txt=None):
    os.makedirs(result_dir, exist_ok=True)

    df = get_df_yolo(gt_dir, ref_txt=ref_txt)
    df.rename(columns={0: 'category', 1: 'center_x', 2:'center_y', 3:'width', 4:'height'}, inplace=True)


    csv_path = os.path.join(result_dir, 'sta.csv')
    df.to_csv(csv_path)
    print('csv save to', csv_path)

    png_shape_path = os.path.join(result_dir, 'object_shape.png')
    sns.jointplot(x='height', y='width', data=df, kind='hex')
    plt.savefig(png_shape_path)
    plt.close()
    print('png save to', png_shape_path)

    png_shapeRate_path = os.path.join(result_dir, 'object_shape_rate.png')
    plt.figure(figsize=(12, 8))
    df['shape_rate'] = (df['width'] / df['height']).round(1)
    df['shape_rate'].value_counts(sort=False, bins=shp_rate_bins).plot(kind='bar', title='images shape rate')
    plt.xticks(rotation=20)
    plt.savefig(png_shapeRate_path)
    plt.close()
    print('png save to', png_shapeRate_path)

    png_pos_start_path = os.path.join(result_dir, 'object_pos_start.png')
    df['pos_sy'] = (df['center_y'] - df['height']*0.5).round(4)
    df['pos_sx'] = (df['center_x'] - df['width']*0.5).round(4)
    g=sns.jointplot(x='pos_sy', y='pos_sx', data=df, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_start_path)
    plt.close()
    print('png save to', png_pos_start_path)

    png_pos_center_path = os.path.join(result_dir, 'object_pos_center.png')
    g=sns.jointplot(x='center_y', y='center_x', data=df, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_center_path)
    plt.close()
    print('png save to', png_pos_center_path)

    png_pos_end_path = os.path.join(result_dir, 'object_pos_end.png')
    df['pos_ey'] = (df['center_y'] + df['height']*0.5).round(4)
    df['pos_ex'] = (df['center_x'] + df['width']*0.5).round(4)
    g=sns.jointplot(x='pos_ey', y='pos_ex', data=df, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_end_path)
    plt.close()
    print('png save to', png_pos_end_path)

    png_cat_path = os.path.join(result_dir, 'object_category.png')
    plt.figure(figsize=(12, 8))
    df['category'].value_counts().sort_index().plot(kind='bar', title='obj category')
    plt.savefig(png_cat_path)
    plt.close()
    print('png save to', png_cat_path)

    png_num_path = os.path.join(result_dir, 'object_number.png')
    plt.figure(figsize=(12, 8))
    df['image'].value_counts().value_counts().sort_index().plot(kind='bar', title='obj number per image')
    plt.xticks(rotation=20)
    plt.savefig(png_num_path)
    plt.close()
    print('png save to', png_num_path)


def get_df_yolo(gt_dir, ref_txt=None, det=True, mdet=False, seg=False):
    pass
    if ref_txt is None:
        gt_list = os.listdir(gt_dir)
    else:
        img_list = pd.read_csv(ref_txt, header=None, index_col=None)[0].tolist()
        gt_list = [Path(img_path).stem+'.txt' for img_path in img_list]

    dfs = []
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ')
        df['image'] = Path(gt_path).stem
        dfs.append(df)
    dfs = pd.concat(dfs)
    if mdet:
        attribute_num = dfs.iat[0, 1]
        cols = list(range(2, 2+attribute_num))
        dfs_attribute = dfs[cols]
        mean_value = dfs_attribute.mean()
        sum_value = dfs_attribute.sum()
        count = (dfs_attribute.sum(axis=1) == 0).sum()
        print(len(gt_list))
        print(len(dfs_attribute), count, len(dfs_attribute)-count)
        print(mean_value)
        print(sum_value)
    return dfs



if __name__ == '__main__':
    pass
    gt_dir = r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\labels"
    val_path = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\val.txt'

    df = get_df_yolo(gt_dir, ref_txt=None, mdet=True)


    # gt_dir = r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\labels"
    # sta_dir = r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\labels_sta"
    # val_path = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\val.txt'
    #
    #
    # yolo_sta(gt_dir, sta_dir)