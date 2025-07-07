import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from isds_tool.PS_data.yolo_mask_crop import myolo_crop_mp

def data_check_compare(csv_path1, csv_path2):
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

    df1 = df1[(df1.index >= 0) & (df1.index <= 7539)]
    df2 = df2[(df2.index >= 0) & (df2.index <= 7539)]

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
    class_file = os.path.join(data_dir, 'class.txt')
    attribute_file = os.path.join(data_dir, 'attribute.yaml')

    check_dir = os.path.join(data_dir, 'data_check_0707')
    check_csv_tjl = os.path.join(check_dir, 'info-check-tjl-update.xlsx')
    check_csv_zys = os.path.join(check_dir, 'info-check-zys-update.xlsx')
    labels_tjl_dir = os.path.join(check_dir, 'labels_tjl')
    labels_zys_dir = os.path.join(check_dir, 'labels_zys')
    images_crop_tjl_dir = os.path.join(check_dir, 'images_crop_tjl')
    images_crop_zys_dir = os.path.join(check_dir, 'images_crop_zys')


    # data_check_compare(check_csv_tjl, check_csv_zys)

    # update_check_result(
    #     labels_dir, labels_tjl_dir, check_csv_tjl, category_list, defect_list,
    # )
    # update_check_result(
    #     labels_dir, labels_zys_dir, check_csv_zys, category_list, defect_list,
    # )

    myolo_crop_mp(images_dir, labels_tjl_dir, images_crop_tjl_dir, class_file,
               attribute_file=attribute_file, seg=True, annotation=True,
               save_method='all', only_defect=True,
               crop_method='with_background_image_shape')

    myolo_crop_mp(images_dir, labels_zys_dir, images_crop_zys_dir, class_file,
               attribute_file=attribute_file, seg=True, annotation=True,
               save_method='all', only_defect=True,
               crop_method='with_background_image_shape')