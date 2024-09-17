import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 示例邻接矩阵（边的权重在0到1之间）
adj_matrix = np.array([
    [0.0, 0.5, 0.2, 0.0],
    [0.5, 0.0, 0.7, 0.3],
    [0.2, 0.7, 0.0, 0.8],
    [0.0, 0.3, 0.8, 0.0]
])

colors = [
    "#FF5733",  # 亮橙色
    "#33FF57",  # 亮绿色
    "#3357FF",  # 亮蓝色
    "#F1C40F",  # 明黄
    "#8E44AD",  # 紫色
    "#E74C3C",  # 红色
    "#2ECC71",  # 绿色
    "#3498DB",  # 蓝色
    "#F39C12",  # 橙色
    "#D35400"   # 深橙色
]


def graph_vis(adj_matrix, labels=False, save_path=None):
    # build the graph
    G = nx.from_numpy_array(adj_matrix)

    edge_colors = [int(c * 100) for c in list(nx.get_edge_attributes(G, 'weight').values())]
    node_colors = [colors[id] for id in G.nodes]
    # edge_weights = nx.get_edge_attributes(G, 'weight')
    # edge_colors = [plt.cm.Blues(weight) for weight in edge_weights.values()]
    # edge_widths = [weight*100 for weight in edge_weights.values()]

    # draw the graph
    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(G, seed=42)  # 选择布局
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx(G, pos,
                     with_labels=False,
                     edge_color='black' if edge_colors is None else edge_colors,
                     node_color='pink' if node_colors is None else node_colors,
                     edge_cmap=plt.cm.Blues,
                     node_size=1000,
                     font_size=20,
                     )

    edge_weights = nx.get_edge_attributes(G, 'weight')
    edge_labels = {}
    for k, v in edge_weights.items():
        if v>0.05:
            edge_labels[k] = '%.2f' % v
        else:
            edge_labels[k] = ''
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, pos, labels=labels)
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()
    plt.close()

def get_labels(df):
    strs = df.columns
    strs_2lines = ['\n'.join(str_s.split('_')) if '_' in str_s else str_s for str_s in strs]
    return {index: value for index, value in enumerate(strs_2lines)}

def matrix_vis(csv_path):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    matrix_array = df.to_numpy()
    rounded_array = np.round(matrix_array, decimals=2)
    np.fill_diagonal(rounded_array, 0)
    print(matrix_array.shape)
    graph_vis(rounded_array, labels=get_labels(df), save_path=csv_path.replace('.csv', '.png'))


if __name__ == '__main__':
    pass
    ROOT_DIR = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c'
    matrix_path1 = os.path.join(ROOT_DIR, 'co_occurrence_matrix1.csv')
    matrix_path2 = os.path.join(ROOT_DIR, 'co_occurrence_matrix2.csv')
    matrix_path3 = os.path.join(ROOT_DIR, 'co_occurrence_matrix3.csv')
    matrix_path4 = os.path.join(ROOT_DIR, 'co_occurrence_matrix4.csv')
    matrix_path5 = os.path.join(ROOT_DIR, 'co_occurrence_matrix5.csv')

    # matrix_vis(matrix_path1)
    # matrix_vis(matrix_path2)
    # matrix_vis(matrix_path3)
    # matrix_vis(matrix_path4)
    matrix_vis(matrix_path5)