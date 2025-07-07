import os
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_sta import dir_shape_sta
from matplotlib import rcParams
# rcParams['font.family'] = 'Times New Roman'
rcParams['font.family'] = 'serif'
shp_rate_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.4, 2.6, 3, 3.5, 4, 5]

# center_x center_y width height
def segmented_bar(df, save_path):
    plt.figure(figsize=(12, 6))
    box_cats = df.columns.to_list()
    assert box_cats[-1] == 'total'
    for cat in box_cats:
        plt.bar(df.index, df[cat], label=cat, color='skyblue')

    # 添加总数标签
    for i, tot in enumerate(df['total']):
        plt.text(i, tot + 5, f'{tot}', ha='center', va='bottom', fontsize=10)

    # 添加标题和标签
    plt.title('Distribution of Signboard Defect by Category', fontsize=16)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # 添加图例
    plt.legend(title='box Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 旋转 x 轴标签，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()



def yolo_sta(gt_dir, result_dir, class_path, attribute_path=None, ref_txt=None, img_dir=None, seg=False):
    os.makedirs(result_dir, exist_ok=True)

    if img_dir is not None:
        img_shape_df = dir_shape_sta(img_dir, os.path.join(result_dir, 'image_shape.png'))
    else:
        img_shape_df = None

    df_class = pd.read_csv(class_path, header=None, index_col=None, names=['class_name'])
    classes = df_class['class_name'].to_list()

    if attribute_path is None:
        df_box, _ = get_df_yolo(gt_dir, ref_txt=ref_txt, classes=classes)
    else:
        df_box, df_attribute = get_df_yolo(gt_dir, ref_txt=ref_txt, classes=classes, attribute_path=attribute_path, mdet=True, seg=seg)

        # region
        csv_path = os.path.join(result_dir, 'sta_attribute.csv')
        df_attribute.to_csv(csv_path)
        print('csv save to', csv_path)

        print('+'*100)
        no_defect_boxes = df_attribute[df_attribute['attribute sum'] == 0].shape[0]
        defect_boxes = df_attribute[df_attribute['attribute sum'] > 0].shape[0]
        print(f"总box数: {len(df_attribute)}")
        print(f"没有缺陷的box数: {no_defect_boxes}")
        print(f"有缺陷的box数: {defect_boxes}")

        png_defect_num_path = os.path.join(result_dir, 'defects_num.png')
        plt.figure(figsize=(10, 8))
        defect_num = df_attribute['attribute sum'].value_counts().sort_index()
        ax = defect_num.plot(kind='bar', title='defects number per box')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.xticks(rotation=0)
        plt.savefig(png_defect_num_path)
        plt.close()


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


        cats = category_defects.drop(index=['attribute sum', 'with attribute'])
        cats = cats.sort_index()
        plt.rcParams.update({
            'font.size': 12,  # 增大字体
            'axes.titlesize': 14,  # 标题字体
            'xtick.labelsize': 11,  # X轴标签字体
            'ytick.labelsize': 11,  # Y轴标签字体
            'legend.fontsize': 10,  # 图例字体
        })
        fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
        colors = ['#6baed6', '#fdae6b', '#74c476']
        bars = cats.plot(
            kind='bar',
            title='distribution of defective signboard',
            ax=ax,
            color=colors,
            width=0.7,
        )
        # category_defects.drop(index=['attribute sum', 'with attribute']).plot(kind='bar', title='defects distribution')
        plt.xticks(rotation=15)
        plt.savefig(png_att_path, bbox_inches='tight', dpi=600)
        plt.close()
        category_defects.to_csv(csv_path.replace('.csv', '_distributions.csv'))
        segmented_bar(category_defects, csv_path.replace('.csv', '_distributions.png'))
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
    plt.xticks(rotation=30)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.savefig(png_cat_path)
    plt.close()
    cat_sta.to_csv(png_cat_path.replace('.png', '.csv'))
    print('+'*100)
    print(cat_sta)
    print('+'*100)
    print('sta result save to', png_cat_path)



    if img_dir is not None:
        df_box = pd.merge(df_box, img_shape_df, on='image', how='left')
        df_box['box_width_pix'] = df_box['width'] * df_box['img_width']
        df_box['box_height_pix'] = df_box['height'] - df_box['img_width']
        png_shape_path = os.path.join(result_dir, 'box_shape_pix.png')
        sns.jointplot(x='box_height_pix', y='box_width_pix', data=df_box, kind='hex')
        plt.savefig(png_shape_path)
        plt.close()
        print('sta result save to', png_shape_path)

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

def poly2xywh(mask):
    mask = np.array([mask[::2], mask[1::2]])
    x_min,y_min = np.min(mask, axis=1)
    x_max,y_max = np.max(mask, axis=1)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]

def get_df_yolo(gt_dir, classes, attribute_path=None, ref_txt=None, mdet=False, seg=False):
    category_dict = {i: name for i, name in enumerate(classes)}

    if ref_txt is None:
        gt_list = os.listdir(gt_dir)
    else:
        img_list = pd.read_csv(ref_txt, header=None, index_col=None)[0].tolist()
        gt_list = [Path(img_path).stem+'.txt' for img_path in img_list]


    if mdet:
        assert attribute_path is not None, 'attribute_path must be provided, which is "%s"'%attribute_path
        with open(attribute_path, 'r') as file:
            attribute_dict = yaml.safe_load(file)['attributes']
        attribute_keys = list(attribute_dict.keys())
        names = ['category'] + ['attribute_len'] + attribute_keys + [ 'center_x', 'center_y', 'width', 'height']
    else:
        names = ['category', 'center_x', 'center_y', 'width', 'height']
    if not seg:
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
        return df_box, df_attribute
    else:
        dfs = []
        for gt_name in tqdm(gt_list):
            gt_path = os.path.join(gt_dir, gt_name)
            df = pd.DataFrame(None, columns=names+['image'])
            with open(gt_path, 'r') as f:
                data = f.readlines()
                if len(data)>100:
                    print(gt_path, len(data))
                for id_line, line in enumerate(data):
                    parts = line.strip().split(' ')
                    category = int(parts[0])
                    image_name = Path(gt_path).stem
                    if mdet:
                        att_len = int(parts[1])
                        atts = list(map(float, parts[2:2+att_len]))
                        polygons = list(map(float, parts[2+att_len:]))
                        xywh = poly2xywh(polygons)
                        df.loc[len(df)] = [category,att_len]+atts+xywh + [image_name]
                    else:
                        polygons = list(map(float, parts[1:]))
                        xywh = poly2xywh(polygons)
                        df.loc[len(df)] = [category]+xywh + [image_name]
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
        return df_box, df_attribute
def info_vis(info_path):
    df = pd.read_csv(info_path, header=0, index_col=0)
    class_counts = df['class_id'].value_counts().sort_index()
    plt.style.use('seaborn')
    plt.figure()

    # 绘制柱状图
    ax1 = plt.subplot()
    sns.barplot(x=class_counts.index, y=class_counts.values,
                palette='viridis', ax=ax1)
    ax1.set_title('类别分布柱状图', fontsize=14, pad=20)
    ax1.set_xlabel('样本数量', fontsize=12)
    ax1.set_ylabel('类别名称', fontsize=12)
    ax1.tick_params(axis='y', labelsize=10)
    plt.show()

if __name__ == '__main__':
    pass
    # yolo_sta(
    #     # img_dir=r"/localnvme/data/billboard/ps_data/psdata735_mseg_c6/images",
    #     img_dir=None,
    #     gt_dir=r"/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels",
    #     result_dir=r"/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels_sta",
    #     class_path=r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/class.txt',
    #     attribute_path=r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/attribute.yaml',
    #     seg=True,
    # )

    data_dir = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6_check0618'
    yolo_sta(
        img_dir=None,
        gt_dir=os.path.join(data_dir, "labels"),
        result_dir=os.path.join(data_dir, "labels_sta"),
        class_path=os.path.join(data_dir, "class.txt"),
        attribute_path=os.path.join(data_dir, "attribute.yaml"),
        seg=True,
    )

    data_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0618'
    yolo_sta(
        img_dir=None,
        gt_dir=os.path.join(data_dir, "labels"),
        result_dir=os.path.join(data_dir, "labels_sta"),
        class_path=os.path.join(data_dir, "class.txt"),
        attribute_path=os.path.join(data_dir, "attribute.yaml"),
        seg=True,
    )

    data_dir = r'/localnvme/data/billboard/fused_data/data1361_mseg_c6_check0618'
    yolo_sta(
        img_dir=None,
        gt_dir=os.path.join(data_dir, "labels"),
        result_dir=os.path.join(data_dir, "labels_sta"),
        class_path=os.path.join(data_dir, "class.txt"),
        attribute_path=os.path.join(data_dir, "attribute.yaml"),
        seg=True,
    )