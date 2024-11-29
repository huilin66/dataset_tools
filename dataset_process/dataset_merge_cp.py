from dataset_merge import *

def search_color_tinydiffer(root_dir, input_name, output_name, temp_name):
    def compare_images(image1, image2):
        mean_value = euclidean(image1.mean(axis=(0, 1)), image2.mean(axis=(0, 1)))
        return mean_value
    df = pd.read_csv(sta_summary_path, header=0, index_col=0)
    for idx, row in df.iterrows():
        data_dir = os.path.join(root_dir, row['name'])
        input_csv_path = os.path.join(data_dir, input_name)
        output_csv_path = os.path.join(data_dir, output_name)
        df_data = pd.read_csv(input_csv_path, header=0, index_col=0)
        df_data['color'] = False
        df_data.to_csv(output_csv_path)


    for i in range(29, 32):
        print(i, len(df))
        src_data_dir = os.path.join(root_dir, df.iloc[i]['name'])
        src_imgs_dir = os.path.join(src_data_dir, 'train', 'images')
        src_data_path = os.path.join(src_data_dir, output_name)
        dst_data_path = os.path.join(src_data_dir, output_name)
        temp_data_path = os.path.join(src_data_dir, temp_name)
        df_src = pd.read_csv(src_data_path, header=0, index_col=0)
        df_src_keep = df_src[df_src['filter'] == False]
        if not os.path.exists(temp_data_path):
            df_temp = pd.DataFrame(1000, index=df_src_keep.index, columns=df_src_keep.index)

            for i in tqdm(range(len(df_src_keep))):
                src_row = df_src_keep.iloc[i]
                src_name = src_row.name
                src_img_path = os.path.join(src_imgs_dir, src_name)
                src_img = io.imread(src_img_path)
                src_img_lab = rgb2lab(src_img)
                for j in range(i, min(i+20, len(df_src_keep))):
                    dst_row = df_src_keep.iloc[j]
                    dst_name = dst_row.name
                    dst_img_path = os.path.join(src_imgs_dir, dst_name)
                    dst_img = io.imread(dst_img_path)
                    dst_img_lab = rgb2lab(dst_img)
                    color_difference = compare_images(src_img_lab, dst_img_lab)
                    df_temp.iloc[i, j] = color_difference
            df_temp.to_csv(temp_data_path)
        else:
            df_temp = pd.read_csv(temp_data_path, header=0, index_col=0)

        for row_name, row in tqdm(df_src_keep.iterrows(), total=len(df_src_keep)):
            record = df_temp.loc[row_name]
            color_tinydiffer_names = record[record < 5].index.tolist()
            for name in color_tinydiffer_names:
                if name == row_name:
                    continue
                else:
                    df_src.loc[name, 'color'] = True
        df_src['filter'] = df_src[['aug', 'swift', 'rotate', 'mirror', 'intersect', 'color']].any(axis=1)
        df_src.to_csv(dst_data_path)


def final_copy(src_dir, dst_dir, ref_name):
    data_list = os.listdir(src_dir)
    shutil.rmtree(dst_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        src_data_dir = os.path.join(src_dir, data_name)
        dst_data_dir = os.path.join(dst_dir, data_name)
        src_imgs_dir = os.path.join(src_data_dir, 'train', 'images')
        dst_imgs_dir = os.path.join(dst_data_dir, 'train', 'images')
        src_labels_dir = os.path.join(src_data_dir, 'train', 'labels')
        dst_labels_dir = os.path.join(dst_data_dir, 'train', 'labels')
        os.makedirs(dst_imgs_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)
        src_data_path = os.path.join(src_data_dir, ref_name)
        df = pd.read_csv(src_data_path, header=0, index_col=0)
        df = df[df['filter'] == False]
        for name, row in tqdm(df.iterrows(), total=len(df)):
            src_img_path = os.path.join(src_imgs_dir, name)
            dst_img_path = os.path.join(dst_imgs_dir, name)
            src_label_path = os.path.join(src_labels_dir, Path(name).stem+'.txt')
            dst_label_path = os.path.join(dst_labels_dir, Path(name).stem+'.txt')
            shutil.copyfile(src_img_path, dst_img_path)
            shutil.copyfile(src_label_path, dst_label_path)


def check_det_label(root_dir):
    def remove_sample(error_info, label_path, image_dir):
        base_name = Path(label_path).stem
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        img_remove = False
        for ext in img_extensions:
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                img_remove = True
                break
        if not img_remove:
            IOError(f'cannot find image for {label_path}')

        os.remove(label_path)
        print(f'{error_info} remove {os.path.basename(label_path)}, {os.path.basename(img_path)}')

    data_list = os.listdir(root_dir)
    for idx, data_name in enumerate(data_list):
        print(data_name, idx, len(data_list))
        data_dir = os.path.join(root_dir, data_name)
        image_dir = os.path.join(data_dir, 'train', 'images')
        label_dir = os.path.join(data_dir, 'train', 'labels_det')

        label_list = os.listdir(label_dir)
        for label_name in label_list:
            label_path = os.path.join(label_dir, label_name)
            if os.path.getsize(label_path) == 0:
                remove_sample(error_info='(empty)', label_path=label_path, image_dir=image_dir)
            else:
                try:
                    df = pd.read_csv(label_path, header=None, index_col=None, sep=' ')
                    if df.shape[1] != 5:
                        remove_sample(error_info=df.shape, label_path=label_path, image_dir=image_dir)
                except Exception as e:
                    remove_sample(error_info=f'(error : {e})', label_path=label_path, image_dir=image_dir)


def dataset_merge(src_dir, dst_dir, sta_summary_path, categories):
    def copy_sample(src_image_path, dst_image_path, src_label_path, dst_label_path, categories_mapping, category):
        df_label = pd.read_csv(src_label_path, header=None, index_col=None, names=['cat_id', 'x', 'y', 'h', 'w'], sep=' ')
        df_label['cat_id'] = df_label['cat_id'].apply(lambda x: categories_mapping[category[x]])
        df_label = df_label[~df_label['cat_id'].isin([categories_mapping['background'], categories_mapping['mix']])]
        if len(df_label) > 0:
            df_label.to_csv(dst_label_path, header=False, index=False, sep=' ')
            shutil.copyfile(src_image_path, dst_image_path)

    categories_mapping = {category: idx for idx, category in enumerate(categories)}

    shutil.rmtree(dst_dir)
    dst_images_dir = os.path.join(dst_dir, 'images')
    dst_labels_dir = os.path.join(dst_dir, 'labels')
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    df_sta = pd.read_csv(sta_summary_path, header=0, index_col=0)
    df_sta['cats'] = df_sta['cats'].apply(ast.literal_eval)
    df_sta['cats_check'] = df_sta['cats_check'].apply(ast.literal_eval)

    # region check
    cats_lengths = df_sta['cats'].apply(len)
    cats_check_lengths = df_sta['cats_check'].apply(len)
    mismatch = df_sta[cats_lengths != cats_check_lengths]
    if len(mismatch) > 0:
        print(mismatch)

    def check_categories(cats_list):
        return all(item in categories for item in cats_list)
    invalid_records = df_sta[~df_sta['cats_check'].apply(check_categories)]
    if len(invalid_records) > 0:
        print(invalid_records)
    # endregion


    for idx, row in df_sta.iterrows():
        category = row['cats_check']
        data_name = row['name']
        print(f'{idx}/{len(df_sta)}, {data_name}, category: {category}')

        src_data_dir = os.path.join(src_dir, data_name)
        src_images_dir = os.path.join(src_data_dir, 'train', 'images')
        src_labels_dir = os.path.join(src_data_dir, 'train', 'labels_det')

        img_list = os.listdir(src_images_dir)
        for img_name in tqdm(img_list):
            src_image_path = os.path.join(src_images_dir, img_name)
            src_label_path = os.path.join(src_labels_dir, Path(img_name).stem+'.txt')
            dst_image_path = os.path.join(dst_images_dir, img_name)
            dst_label_path = os.path.join(dst_labels_dir, Path(img_name).stem+'.txt')

            copy_sample(src_image_path, dst_image_path, src_label_path, dst_label_path, categories_mapping, category)



if __name__ == '__main__':
    pass
    data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo'
    dst_data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo_final'
    merge_data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo_merge'
    sta_summary_path = r'E:\data\2024_defect\2024_defect_pure_yolo_sta\sta_summary.csv'
    categories = ['background',
                  'crack', 'hole', 'blister', 'delamination', 'peeling',
                  'spalling', 'mold', 'corrosion', 'condensation', 'stain',
                  'vegetation', 'mix']


    # search_color_tinydiffer(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv', 'data_rmaug_rmshift_rmrotate_rmmirror_rminter_rmcolor.csv', 'color_difference.csv')

    # get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter.csv')
    # get_remained_img(data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter_rmcolor.csv')

    # final_copy(data_dir, dst_data_dir, 'data_rmaug_rmshift_rmrotate_rmmirror_rminter_rmcolor.csv')

    # convert_seg2det(dst_data_dir, sta_summary_path)

    # check_det_label(dst_data_dir)

    # dataset_vis(dst_data_dir, sta_summary_path)

    dataset_merge(dst_data_dir, merge_data_dir, sta_summary_path, categories)
