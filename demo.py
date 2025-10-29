import os.path

import pandas
import pandas as pd


def get_img_list(input_csv_path):
    df = pd.read_csv(input_csv_path, index_col=None, names=['file_name'])
    img_list = df['file_name'].to_list()
    img_list = [os.path.basename(file_path) for file_path in img_list]
    return img_list

if __name__ == '__main__':
    pass
    train_path = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/train.txt'
    img_list_train = get_img_list(train_path)
    val_path = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val.txt'
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