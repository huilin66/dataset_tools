import os, yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

def get_com(gt_dir, attribute_file, save_path, filter_background=False, norm=False):
    with open(attribute_file, 'r') as file:
        attribute_dict = yaml.safe_load(file)['attributes']

    gt_list = os.listdir(gt_dir)
    dfs = []
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ')
        df['image'] = Path(gt_path).stem
        dfs.append(df)
    dfs = pd.concat(dfs)

    attribute_num = dfs.iat[0, 1]
    cols = list(range(2, 2+attribute_num))
    dfs_attribute = dfs[cols]

    category_matrix = dfs_attribute.to_numpy()
    if filter_background:
        category_matrix = category_matrix[category_matrix.sum(axis=1) != 0]

    co_occurrence_matrix = np.dot(category_matrix.T, category_matrix)
    co_occurrence_probability_matrix = co_occurrence_matrix / category_matrix.shape[0]

    if norm:
        diagonal_values = np.diag(co_occurrence_probability_matrix)
        co_occurrence_probability_matrix /= diagonal_values[:, np.newaxis]


    df = pd.DataFrame(co_occurrence_probability_matrix, columns=list(attribute_dict.keys()), index=list(attribute_dict.keys()))
    df.to_csv(save_path)
    print(co_occurrence_probability_matrix)



if __name__ == '__main__':
    pass
    # get_com(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\labels',
    #         r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\attribute.yaml',
    #         r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\co_occurrence_matrix.csv',
    #         filter_background=True,
    #         norm=True
    #         )
    # get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\labels',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\attribute.yaml',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10\co_occurrence_matrix3.csv',
    #         filter_background=True,
    #         norm=True
    #         )

    get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix1.csv',
            filter_background=False,
            norm=False
            )
    get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix2.csv',
            filter_background=True,
            norm=False
            )
    get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
            r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix3.csv',
            filter_background=True,
            norm=True
            )