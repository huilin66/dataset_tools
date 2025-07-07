import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def update_check_result(src_labels_dir, dst_labels_dir, check_csv_path, category_list, defect_list, level_list):
    df_check = pd.read_excel(check_csv_path, header=0, index_col=0)
    df_check['object_name_stem'] = df_check['object_name'].apply(lambda x: Path(x).stem)
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
            result_row = df_check.loc[df_check['object_name_stem'] == object_name_stem]
            if len(result_row) != 1:
                print(object_name_stem, 'error!')
            category_name = str(result_row['category'].values[0])
            category_id = category_list.index(category_name)
            line_parts[0] = str(category_id)
            for idx, defect_name in enumerate(defect_list):
                defect_value = int(result_row[defect_name].values[0])

                # defect_level = level_list[defect_value]
                line_parts[2+idx] = str(defect_value)
            line_update = ' '.join(line_parts)+'\n'
            new_lines.append(line_update)
        with open(dst_label_path, 'w') as fw:
            fw.writelines(new_lines)






if __name__ == '__main__':
    pass
    # src_labels_dir = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6/labels'
    # dst_labels_dir = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6_check0618/labels'
    # check_csv_path = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6_check0618/info_PS_check_zys_result0618.xlsx'

    src_data_dir = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6_check0618'
    src_class_file = os.path.join(src_data_dir, 'class.txt')
    src_attribute_file = os.path.join(src_data_dir, 'attribute.yaml')
    src_labels_dir = os.path.join(src_data_dir, 'labels')

    dst_data_dir = r'/localnvme/data/billboard/ps_data/psdata735_mseg_c6_check0624/'
    dst_labels_dir = os.path.join(dst_data_dir, 'labels')

    check_csv_path = os.path.join(dst_data_dir, 'info_PS_check_tjl_result0624.xlsx')
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
    update_check_result(src_labels_dir, dst_labels_dir, check_csv_path, category_list, defect_list, level_list)