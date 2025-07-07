import os, yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

'''
6种两两类别生成贡献概率矩阵的方法
1：直接统计得到
2：统计后，剔除不包括任何attribute的negative样本
3：对角线归一化，每一列除以对角线的值
4：3的转置，对角线归一化，每一行除以对角线的值
5：（3+4）/2
6：提取对角线平方根后，先按行归一化，再按列归一化
'''

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

def cal_com(com_src_path, com_dst_path):
    df_com = pd.read_csv(com_src_path, header=0, index_col=0)
    array_com = df_com.to_numpy()
    array_com_t = array_com.T
    dst_com = (array_com + array_com_t) * 0.5
    df_dst_com = pd.DataFrame(dst_com, index=df_com.index, columns=df_com.columns)
    df_dst_com.to_csv(com_dst_path)
    print('save to', com_dst_path)

def get_com_t(com_src_path, com_dst_path):
    df_com = pd.read_csv(com_src_path, header=0, index_col=0)
    df_com_t = df_com.T
    df_com_t.to_csv(com_dst_path)
    print('save to', com_dst_path)

def get_cross_norm(com_src_path, com_dst_path):
    df_com = pd.read_csv(com_src_path, header=0, index_col=0)
    diagonal_values = np.diag(df_com)
    sqrt_diagonal = np.sqrt(diagonal_values)


    df_normalized = df_com.copy()
    df_normalized = df_normalized.div(sqrt_diagonal, axis=0)
    df_normalized = df_normalized.div(sqrt_diagonal, axis=1)
    df_normalized.to_csv(com_dst_path)
    print('save to', com_dst_path)


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

    # get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix1.csv',
    #         filter_background=False,
    #         norm=False
    #         )
    # get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix2.csv',
    #         filter_background=True,
    #         norm=False
    #         )
    # get_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\attribute.yaml',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix3.csv',
    #         filter_background=True,
    #         norm=True
    #         )
    # get_com_t(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix3.csv',
    #           r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix4.csv',)
    # cal_com(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix3.csv',
    #         r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix5.csv',)

    get_cross_norm(r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix1.csv',
                   r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\co_occurrence_matrix6.csv',)