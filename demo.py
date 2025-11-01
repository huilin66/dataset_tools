import os.path
import shutil

import pandas
import pandas as pd


def get_img_list(input_csv_path):
    df = pd.read_csv(input_csv_path, index_col=None, names=['file_name'])
    img_list = df['file_name'].to_list()
    img_list = [os.path.basename(file_path) for file_path in img_list]
    return img_list



if __name__ == '__main__':
    pass
    train_path = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1031_v4/train_test.txt'
    img_list_train = get_img_list(train_path)
    val_path = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1031_v4/val_test.txt'
    img_list_val = get_img_list(val_path)
    test_path = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_1021/val.txt'
    img_list_test = get_img_list(test_path)
    common_list1 = []
    for img_path in img_list_train:
        if img_path in img_list_val:
            common_list1.append(img_path)
    print(common_list1)

    common_list2 = []
    for img_path in img_list_train:
        if img_path in img_list_test:
            common_list2.append(img_path)
    print(common_list2)

    common_list3 = []
    for img_path in img_list_val:
        if img_path in img_list_test:
            common_list3.append(img_path)
    print(common_list3)

    # root_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine/result_analysis/keep/pred_no_label_background'
    # sub_dir_list = os.listdir(root_dir)
    # for sub_dir in sub_dir_list:
    #     sub_dir_path = os.path.join(root_dir, sub_dir)
    #     for filename in os.listdir(sub_dir_path):
    #         input_file_path = os.path.join(sub_dir_path, filename)
    #         output_file_path = os.path.join(root_dir, filename)
    #         shutil.move(input_file_path, output_file_path)