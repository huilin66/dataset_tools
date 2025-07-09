import os, cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from isds_tool.PS_data.yolo_mask_crop import myolo_crop_mp, myolo_crop

def image_read(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
def image_save(img_path, img):
    cv2.imencode('.png', img)[1].tofile(img_path)

def data_check_compare(csv_path1, csv_path2, csv_path3):
    columns_to_compare = [
        "category_x",	
        "deformation",	
        "broken",	
        "abandonment",	
        "corrosion",	
        "object_name"
        ]
    df1 = pd.read_excel(csv_path1, header=0, index_col=0)
    df2 = pd.read_excel(csv_path2, header=0, index_col=0)
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    assert df1.shape == df2.shape, print(f'{os.path.basename(csv_path1)} with {df1.shape}, while {os.path.basename(csv_path2)} with {df2.shape}!')

    # df1 = df1[(df1.index >= 0) & (df1.index <= 7539)]
    # df2 = df2[(df2.index >= 0) & (df2.index <= 7539)]

    diff_mask = (df1[columns_to_compare] != df2[columns_to_compare])
    print(diff_mask.shape)


    # Find rows where any of the specified columns differ
    rows_with_differences = diff_mask.any(axis=1)
    print(rows_with_differences.shape)
    print("Rows with differences in specified columns:")
    print("df1 differences:")
    print(df1.loc[rows_with_differences, columns_to_compare])
    print("df2 differences:")
    print(df2.loc[rows_with_differences, columns_to_compare])

    df2['different'] = rows_with_differences.astype(int)
    # df2.to_excel(csv_path3)

    matching_image_names = df1.loc[rows_with_differences, "object_name"].to_list()
    return matching_image_names


def update_check_result(src_labels_dir, dst_labels_dir, check_csv_path, category_list, defect_list):
    df_check = pd.read_excel(check_csv_path, header=0, index_col=0)
    os.makedirs(dst_labels_dir, exist_ok=True)
    labels_list = os.listdir(src_labels_dir)
    for label_name in tqdm(labels_list):
        src_label_path = os.path.join(src_labels_dir, label_name)
        dst_label_path = os.path.join(dst_labels_dir, label_name)
        with open(src_label_path, 'r') as fr:
            lines = fr.readlines()

        new_lines = []
        for idx, line in enumerate(lines):
            line_parts = line.strip().split()
            object_name_stem = f'{Path(label_name).stem}_{idx}'
            result_row = df_check.loc[df_check['object_name'] == object_name_stem]
            if len(result_row) != 1:
                print(object_name_stem, 'error!')
            category_name = str(result_row['category_x'].values[0])
            category_id = category_list.index(category_name)
            line_parts[0] = str(category_id)
            for idx, defect_name in enumerate(defect_list):
                defect_value = int(result_row[defect_name].values[0])
                line_parts[2+idx] = str(defect_value)
            line_update = ' '.join(line_parts)+'\n'
            new_lines.append(line_update)
        with open(dst_label_path, 'w') as fw:
            fw.writelines(new_lines)


def img_cat(img_list, dst_path):
    imgs = [image_read(img_path) for img_path in img_list]
    img = np.concatenate(imgs, axis=1)
    image_save(dst_path, img)

def img_cat2(img_path1, img_path2, dst_path):
    img1 = image_read(img_path1)
    img2 = image_read(img_path2)
    img = np.concatenate([img1, img2], axis=1)
    image_save(dst_path, img)

def images_cat(image_list, input_dir1, input_dir2, output_dir, ref_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, img_name in enumerate(tqdm(image_list)):
        if idx<45:
            continue
        input_path1 = os.path.join(input_dir1, img_name+'.png') if os.path.exists(os.path.join(input_dir1, img_name+'.png')) else os.path.join(input_dir1, img_name+'.jpg')
        input_path2 = os.path.join(input_dir2, img_name+'.png') if os.path.exists(os.path.join(input_dir2, img_name+'.png')) else os.path.join(input_dir2, img_name+'.jpg')
        output_path = os.path.join(output_dir, img_name+'.png')
        img_cat2(input_path1, input_path2, output_path)

if __name__ == '__main__':
    pass

    defect_list = ['deformation', 'broken', 'abandonment', 'corrosion']
    level_list = ['no', 'medium', 'high']
    category_list = ['wall frame',
                     'wall display',
                     'projecting frame',
                     'projecting display',
                     'hanging frame',
                     'hanging display',
                     'other'
                     ]

    data_dir = r'E:\data\202502_signboard\data_annotation\dataset\data1422'
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    images_crop_dir = os.path.join(data_dir, 'images_crop')
    class_file = os.path.join(data_dir, 'class.txt')
    attribute_file = os.path.join(data_dir, 'attribute.yaml')

    check_dir = os.path.join(data_dir, 'data_check_0707')
    check_csv_tjl = os.path.join(check_dir, 'info-check-tjl-update.xlsx')
    check_csv_zys = os.path.join(check_dir, 'info-check-zys-update.xlsx')
    check_csv_zhl = os.path.join(check_dir, 'info-check-zhl-update.xlsx')
    labels_tjl_dir = os.path.join(check_dir, 'labels_tjl')
    labels_zys_dir = os.path.join(check_dir, 'labels_zys')
    labels_zhl_dir = os.path.join(check_dir, 'labels_zhl')
    images_crop_tjl_dir = os.path.join(check_dir, 'images_crop_tjl')
    images_crop_zys_dir = os.path.join(check_dir, 'images_crop_zys')
    images_crop_zhl_dir = os.path.join(check_dir, 'images_crop_zhl')
    images_crop_compare_dir = os.path.join(check_dir, 'images_crop_compare')



    # update_check_result(
    #     labels_dir, labels_tjl_dir, check_csv_tjl, category_list, defect_list,
    # )
    # update_check_result(
    #     labels_dir, labels_zys_dir, check_csv_zys, category_list, defect_list,
    # )

    # image_name_list = data_check_compare(check_csv_tjl, check_csv_zys, check_csv_zhl)

    # myolo_crop(images_dir, labels_tjl_dir, images_crop_tjl_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=True,
    #            save_method='all', ref_list=image_name_list,
    #            crop_method='with_background_image_shape')
    # myolo_crop(images_dir, labels_zys_dir, images_crop_zys_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=True,
    #            save_method='all', ref_list=image_name_list,
    #            crop_method='with_background_image_shape')

    # images_cat(image_name_list, images_crop_tjl_dir, images_crop_zys_dir, images_crop_compare_dir, images_crop_dir)

    update_check_result(
        labels_dir, labels_zhl_dir, check_csv_zhl, category_list, defect_list,
    )