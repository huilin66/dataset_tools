import os
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


shp_rate_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.4, 2.6, 3, 3.5, 4, 5]

# center_x center_y width height
def yolo_sta(gt_dir, result_dir, class_path, attribute_path=None, ref_txt=None):
    os.makedirs(result_dir, exist_ok=True)

    df_class = pd.read_csv(class_path, header=None, index_col=None, names=['class_name'])
    classes = df_class['class_name'].to_list()

    if attribute_path is None:
        df_box, _ = get_df_yolo(gt_dir, ref_txt=ref_txt, classes=classes)
    else:
        df_box, df_attribute = get_df_yolo(gt_dir, ref_txt=ref_txt, classes=classes, attribute_path=attribute_path, mdet=True)

        # region
        csv_path = os.path.join(result_dir, 'sta_attribute.csv')
        df_attribute.to_csv(csv_path)
        print('csv save to', csv_path)


        image_count = df_attribute['image'].nunique()
        count = (df_attribute['attribute sum'] == 0).sum()
        print('+'*100)
        no_defect_boxes = df_attribute[df_attribute['attribute sum'] == 0].shape[0]
        defect_boxes = df_attribute[df_attribute['attribute sum'] > 0].shape[0]
        print(f"总box数: {len(df_attribute)}")
        print(f"没有缺陷的box数: {no_defect_boxes}")
        print(f"有缺陷的box数: {defect_boxes}")

        unique_image_count = df_attribute['image'].nunique()
        defect_images = df_attribute.groupby('image')['attribute sum'].sum()
        no_defect_images = defect_images[defect_images == 0].count()
        defect_images = defect_images[defect_images > 0].count()
        print(f"总image数: {unique_image_count}")
        print(f"没有缺陷的image数: {no_defect_images}")
        print(f"有缺陷的image数: {defect_images}")
        png_att_path = os.path.join(result_dir, 'attribute_num.png')
        category_defects = df_attribute.groupby('category').sum().drop(columns=['image'])
        total_defects = category_defects.sum(axis=0)
        category_defects.loc['total'] = total_defects
        category_defects = category_defects.T

        plt.figure(figsize=(12, 8))
        category_defects.drop(index=['attribute sum', 'with attribute']).plot(kind='bar')
        plt.xticks(rotation=15)
        plt.savefig(png_att_path)
        plt.close()
        print('+'*100)
        print(category_defects)
        print('+'*100)
        print('sta result save to', png_att_path)
        # endregion

    # region: box sta result
    csv_path = os.path.join(result_dir, 'sta_box.csv')
    df_box.to_csv(csv_path)
    print('csv save to', csv_path)

    png_cat_path = os.path.join(result_dir, 'box_category.png')
    plt.figure(figsize=(12, 8))
    cat_sta = df_box['category'].value_counts().sort_index()
    ax = cat_sta.plot(kind='bar', title='obj category')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig(png_cat_path)
    plt.close()
    print('+'*100)
    print(cat_sta)
    print('+'*100)
    print('sta result save to', png_cat_path)

    png_shape_path = os.path.join(result_dir, 'box_shape.png')
    sns.jointplot(x='height', y='width', data=df_box, kind='hex')
    plt.savefig(png_shape_path)
    plt.close()
    print('sta result save to', png_shape_path)

    png_shapeRate_path = os.path.join(result_dir, 'box_shape_rate.png')
    plt.figure(figsize=(12, 8))
    df_box['shape_rate'] = (df_box['width'] / df_box['height']).round(1)
    df_box['shape_rate'].value_counts(sort=False, bins=shp_rate_bins).plot(kind='bar', title='images shape rate')
    plt.xticks(rotation=20)
    plt.savefig(png_shapeRate_path)
    plt.close()
    print('sta result save to', png_shapeRate_path)

    png_pos_start_path = os.path.join(result_dir, 'box_pos_start.png')
    df_box['pos_sy'] = (df_box['center_y'] - df_box['height']*0.5).round(4)
    df_box['pos_sx'] = (df_box['center_x'] - df_box['width']*0.5).round(4)
    g=sns.jointplot(x='pos_sx', y='pos_sy', data=df_box, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_start_path)
    plt.close()
    print('sta result save to', png_pos_start_path)

    png_pos_center_path = os.path.join(result_dir, 'box_pos_center.png')
    g=sns.jointplot(x='center_x', y='center_y', data=df_box, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_center_path)
    plt.close()
    print('sta result save to', png_pos_center_path)

    png_pos_end_path = os.path.join(result_dir, 'box_pos_end.png')
    df_box['pos_ey'] = (df_box['center_y'] + df_box['height']*0.5).round(4)
    df_box['pos_ex'] = (df_box['center_x'] + df_box['width']*0.5).round(4)
    g=sns.jointplot(x='pos_ex', y='pos_ey', data=df_box, kind='hex')
    g.ax_joint.invert_yaxis()
    plt.savefig(png_pos_end_path)
    plt.close()
    print('sta result save to', png_pos_end_path)

    png_num_path = os.path.join(result_dir, 'box_number.png')
    plt.figure(figsize=(12, 8))
    df_box['image'].value_counts().value_counts().sort_index().plot(kind='bar', title='box number per image')
    plt.xticks(rotation=20)
    plt.savefig(png_num_path)
    plt.close()
    print('sta result save to', png_num_path)
    # endregion

def get_df_yolo(gt_dir, classes, attribute_path=None, ref_txt=None, mdet=False, seg=False):
    category_dict = {i: name for i, name in enumerate(classes)}

    if ref_txt is None:
        gt_list = os.listdir(gt_dir)
    else:
        img_list = pd.read_csv(ref_txt, header=None, index_col=None)[0].tolist()
        gt_list = [Path(img_path).stem+'.txt' for img_path in img_list]

    if not seg:
        if mdet:
            assert attribute_path is not None, 'attribute_path must be provided, which is "%s"'%attribute_path
            with open(attribute_path, 'r') as file:
                attribute_dict = yaml.safe_load(file)['attributes']
            attribute_keys = list(attribute_dict.keys())
            names = ['category'] + ['attribute_len'] + attribute_keys + [ 'center_x', 'center_y', 'width', 'height']
        else:
            names = ['category', 'center_x', 'center_y', 'width', 'height']

        dfs = []
        for gt_name in tqdm(gt_list):
            gt_path = os.path.join(gt_dir, gt_name)
            df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ', names=names)
            df['image'] = Path(gt_path).stem
            dfs.append(df)
        dfs = pd.concat(dfs)
        dfs['category'] = dfs['category'].map(category_dict)
        df_box = dfs[['category', 'center_x', 'center_y', 'width', 'height', 'image']].copy()
        if mdet:
            df_attribute = dfs[['category', 'image']+attribute_keys].copy()
            df_attribute['attribute sum'] = df_attribute.iloc[:, 2:].sum(axis=1)
            df_attribute['with attribute'] = df_attribute['attribute sum'].apply(lambda x: 0 if x == 0 else 1)
        else:
            df_attribute = None
        # if mdet:
        #     attribute_num = dfs.iat[0, 1]
        #     cols = list(range(2, 2+attribute_num))
        #     dfs_attribute = dfs[cols]
        #     mean_value = dfs_attribute.mean()
        #     sum_value = dfs_attribute.sum()
        #     count = (dfs_attribute.sum(axis=1) == 0).sum()
        #     print(len(gt_list))
        #     print(len(dfs_attribute), count, len(dfs_attribute)-count)
        #     print(mean_value)
        #     print(sum_value)
        return df_box, df_attribute



if __name__ == '__main__':
    pass
    # yolo_sta(
    #     gt_dir=r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\labels",
    #     result_dir=r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\labels_sta",
    #     class_path=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\class.txt'
    #     # val_path = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\val.txt',
    # )

    yolo_sta(
        gt_dir=r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\labels",
        result_dir=r"E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\labels_sta",
        class_path=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\class.txt',
        attribute_path=r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\attribute.yaml',
        # val_path = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_det\val.txt',
    )