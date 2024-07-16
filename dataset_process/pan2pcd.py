import os
import cv2
import copy
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
import numpy as np
from skimage import io
from tqdm import tqdm
from pathlib import Path

MATRIX = np.array([
    [1.000, 0.000, 0.000, 836538.562, ],
    [0.000, 1.000, 0.000, 818403.562, ],
    [0.000, 0.000, 1.000, 13.250],
    [0.000, 0.000, 0.000, 1.000]
])

# region image tools

def img_show(img_path):
    '''
    可视化图片，便于手动选择框
    :param img_path:
    :return:
    '''
    img = io.imread(img_path)
    plt.imshow(img)
    plt.show()


def img_box_vis(panorama_path, bbox, box_vis_path):
    '''
    将bbox可视化到图像中
    :param panorama_path:
    :param bbox:
    :param box_vis_path:
    :return:
    '''
    os.makedirs(os.path.dirname(box_vis_path), exist_ok=True)
    pan_img = io.imread(panorama_path)
    cv2.rectangle(pan_img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 255, 0), 2)
    io.imsave(box_vis_path, pan_img)

def img_boxes_vis(panorama_path, bboxes, box_vis_path):
    '''
    将bboxes可视化到图像中
    :param panorama_path:
    :param bbox:
    :param box_vis_path:
    :return:
    '''
    os.makedirs(os.path.dirname(box_vis_path), exist_ok=True)
    pan_img = io.imread(panorama_path)
    for bbox in bboxes:
        cv2.rectangle(pan_img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (0, 255, 0), 2)
    io.imsave(box_vis_path, pan_img)

# endregion

# region point cloud tools

def read_trajectory(pcd_file):
    '''
    读取特定的轨迹文件
    :param pcd_file:
    :return:
    '''
    df = pd.DataFrame(None, columns=['x','y','z','index','R00','R01','R02','R10','R11','R12','R20','R21','R22'])
    header = ""
    data_flag = False
    with open(pcd_file, 'rb') as f:
        for line in f.readlines():
            line = line.decode('utf-8').strip('\r\n')
            if not data_flag:
                header += line
                if line.startswith("DATA"):
                    data_flag = True
            else:
                row = line.split(' ')
                df.loc[len(df)] = row
    return header, df


def get_posbytrajectory(panorama_path, trajectory_path):
    '''
    从轨迹数据中，获取全景图像信息，返回相机3d坐标、旋转矩阵
    :param panorama_path:
    :param trajectory_path:
    :return:
    '''
    header, df_data = read_trajectory(trajectory_path)

    pan_name = os.path.basename(panorama_path)
    row = df_data[df_data['index'] == pan_name.strip('.jpg')]

    pos = np.array([float(row['x']), float(row['y']), float(row['z']), ])
    rotate_matrix = np.array([float(row['R00']), float(row['R01']), float(row['R02']),
                              float(row['R10']), float(row['R11']), float(row['R12']),
                              float(row['R20']), float(row['R21']), float(row['R22']),
                              ]).reshape(3, 3)

    print('center:', pos)
    print('director', rotate_matrix)
    return pos, rotate_matrix


def get_pcd(pcd_path, center_pos, radius=None, save_path=None):
    '''
    根据相机3d坐标，获取一定范围内的点云，减少数据处理量
    :param pcd_path:
    :param center_pos:
    :param radius:
    :param save_path:
    :return:
    '''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pcd = o3d.io.read_point_cloud(pcd_path)
    if radius is None:
        return pcd

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    distances = np.sqrt(np.sum((points - center_pos)**2, axis=1))

    local_points = points[distances < radius]
    local_colors = colors[distances < radius]
    print('radius %d, points : %d --> %d'%(radius, len(points), len(local_points)))

    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(local_points)
    local_pcd.colors = o3d.utility.Vector3dVector(local_colors)
    o3d.io.write_point_cloud(save_path, local_pcd)
    return local_pcd


def thicken_image(image):
    '''
    加密图像，将点云投影的系数图像稠密化
    :param image:
    :return:
    '''
    from scipy.interpolate import griddata
    # 获取已知像素的索引和RGB值
    known_indices = np.where(~np.isnan(image).any(axis=2))
    known_points = np.column_stack((known_indices[0], known_indices[1]))
    known_values = image[known_indices]

    # 获取未知像素的索引
    unknown_indices = np.where(np.isnan(image).any(axis=2))
    unknown_points = np.column_stack((unknown_indices[0], unknown_indices[1]))

    # 执行插值
    interpolated_values = griddata(known_points, known_values, unknown_points, method='linear')

    # 更新图像矩阵
    image[unknown_indices] = interpolated_values
    return image


def map_pcd2pan(pcd_points, center_pos, H, W, director=None, save_path=None, vis_path=None, depth_path=None, thicken=False):
    '''
    将点云映射到全景图像，生成全景图片到点云坐标的映射文件
    :param pcd_points:
    :param center_pos:
    :param H:
    :param W:
    :param director:
    :param save_path:
    :param vis_path:
    :param depth_path:
    :param thicken:
    :return:
    '''
    def get_rt_matrix(rot, pos):
        rt_matrix = np.zeros((4,4))
        rt_matrix[3, 3] = 1
        rt_matrix[:3, :3] = rot
        rt_matrix[:3, 3]  = pos.T
        rt_matrix = np.linalg.inv(rt_matrix)
        return rt_matrix

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(vis_path), exist_ok=True)

    assert isinstance(pcd_points, o3d.geometry.PointCloud), 'input data must be PointCloud'
    TM = get_rt_matrix(director, center_pos)
    print(TM)
    points_rel = copy.deepcopy(pcd_points).transform(TM)
    pcd_colors = np.array(points_rel.colors)
    points_rel = np.array(points_rel.points)
    pcd_points = np.array(pcd_points.points)

    # region convert pcd 2 pan
    x, y, z = np.split(points_rel, 3, axis=1)
    r = np.sqrt(x*x+y*y+z*z)

    alpha = np.arctan2(y, x)
    omega = np.arcsin(z/r)
    print('alpha:', np.min(alpha), np.max(alpha))
    print('omega:', np.min(omega), np.max(omega))
    # alpha: -pi~0 -> 0~pi; 0~pi -> -pi~0
    alpha += np.pi
    alpha = np.where(alpha > np.pi, alpha-2 * np.pi, alpha)
    print('alpha:', np.min(alpha), np.max(alpha))

    u = 0.5 - alpha/(2*np.pi)
    v = 0.5 - omega/np.pi
    print('u:', np.min(u), np.max(u))
    print('v:', np.min(v), np.max(v))
    m = np.round(u*W)
    n = np.round(v*H)
    print('m:', np.min(m), np.max(m))
    print('n:', np.min(n), np.max(n))
    pos2d_points = np.concatenate([m, n], axis=1)
    # endregion

    #region map the conversion into 2d array
    map_array = np.full((H, W, 3), np.nan)
    dis_array = np.ones((H,W,1))*255
    if vis_path is not None:
        # color_array = np.ones((H,W,3))*255
        color_array = np.full((H, W, 3), np.nan)

    x2d, y2d = np.split(pos2d_points, 2, axis=1)
    x2d_int, y2d_int = np.round(x2d), np.round(y2d)
    x3d, y3d, z3d = np.split(pcd_points, 3, axis=1)
    dis3d = np.sqrt(np.square(center_pos[0] - x3d) +
                      np.square(center_pos[1] - y3d) +
                      np.square(center_pos[2] - z3d)
                      )
    for idx in tqdm(range(pcd_points.shape[0])):
        x_int, y_int, dis, point_3d = int(x2d_int[idx]), int(y2d_int[idx]), dis3d[idx], pcd_points[idx]
        if x_int>=W or y_int>=H:
            continue
        if dis<dis_array[y_int, x_int]:
            map_array[y_int, x_int] = point_3d
            dis_array[y_int, x_int] = dis
            if vis_path is not None:
                color_array[y_int, x_int] = pcd_colors[idx]

    if thicken:
        map_nan_num1 = np.sum(np.isnan(map_array).any(axis=2))
        map_array = thicken_image(map_array)
        map_nan_num2 = np.sum(np.isnan(map_array).any(axis=2))

        dis_array[dis_array==255]=np.nan
        depth_nan_num1 = np.sum(np.isnan(dis_array))
        dis_array = thicken_image(dis_array)
        depth_nan_num2 = np.sum(np.isnan(dis_array))

        color_nan_num1 = np.sum(np.isnan(color_array).any(axis=2))
        color_array = thicken_image(color_array)
        color_nan_num2 = np.sum(np.isnan(color_array).any(axis=2))

        print('map nan num: %d --> %d' % (map_nan_num1, map_nan_num2))
        print('depth nan num: %d --> %d' % (depth_nan_num1, depth_nan_num2))
        print('color nan num: %d --> %d' % (color_nan_num1, color_nan_num2))

    #endregion
    if save_path is not None:
        np.save(save_path, map_array)
    if vis_path is not None:
        io.imsave(vis_path, np.array(color_array*255, dtype=np.uint8))
    if depth_path is not None:
        np.save(depth_path, dis_array)
    return map_array


def create_3d_bbox_edges(point_lt, point_rb, num_points_per_edge=10):
    '''
    绘制出3d bbox的点云
    :param point_lt:
    :param point_rb:
    :param num_points_per_edge:
    :return:
    '''
    # 计算宽度、高度和深度
    width = abs(point_lt[0] - point_rb[0])
    height = abs(point_lt[1] - point_rb[1])
    depth = abs(point_lt[2] - point_rb[2])
    point_min = [min(point_lt[0], point_rb[0]),
                 min(point_lt[1], point_rb[1]),
                 min(point_lt[2], point_rb[2])]

    # 创建3D目标检测框的8个顶点
    vertices = [point_lt,
                [point_min[0] + width, point_min[1], point_min[2]],
                [point_min[0] + width, point_min[1] + height, point_min[2]],
                [point_min[0], point_min[1] + height, point_min[2]],
                [point_min[0], point_min[1], point_min[2] + depth],
                [point_min[0] + width, point_min[1], point_min[2] + depth],
                [point_min[0] + width, point_min[1] + height, point_min[2] + depth],
                [point_min[0], point_min[1] + height, point_min[2] + depth]]
    # 创建3D目标检测框的12条边
    edges = [[vertices[i], vertices[j]] for i, j in
             [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]]

    # 使用线性插值在每条边上生成10个点
    edge_points = []
    for edge in edges:
        edge_points.extend(np.linspace(edge[0], edge[1], num_points_per_edge).tolist())

    return edge_points


def box_pan2pcd_measure(bbox_pan, pan2pcd_path, vis_path=None):
    '''
    根据 全景图片到点云坐标的映射文件，将bbox转换为3d bbox
    :param bbox_pan:
    :param pan2pcd_path:
    :param vis_path:
    :return:
    '''
    box_pan_lt, box_pan_rb = bbox_pan
    pan2pcd = np.load(pan2pcd_path)
    print(np.max(pan2pcd), np.min(pan2pcd))
    count_non_white = np.sum(np.any(pan2pcd != [np.nan, np.nan, np.nan], axis=-1))
    count_white = np.sum(np.any(pan2pcd == [np.nan, np.nan, np.nan], axis=-1))
    print(count_non_white, count_white)
    box_pcd_lt = pan2pcd[box_pan_lt[1], box_pan_lt[0]]
    box_pcd_rb = pan2pcd[box_pan_rb[1], box_pan_rb[0]]
    print('2D(%d,%d) --> 3D(%g, %g, %g)' % (box_pan_lt[0], box_pan_lt[1], box_pcd_lt[0], box_pcd_lt[1], box_pcd_lt[2]))
    print('2D(%d,%d) --> 3D(%g, %g, %g)' % (box_pan_rb[0], box_pan_rb[1], box_pcd_rb[0], box_pcd_rb[1], box_pcd_rb[2]))

    print('%.2fm, %.2fm, %.2fm'%(abs(box_pcd_rb[0]-box_pcd_lt[0]), abs(box_pcd_rb[1]-box_pcd_lt[1]), abs(box_pcd_rb[2]-box_pcd_lt[2])))
    print('height(center): %g'%((box_pcd_lt[2]+box_pcd_rb[2])/2))
    if vis_path is not None:
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        vis_points = create_3d_bbox_edges(box_pcd_lt, box_pcd_rb, num_points_per_edge=10)

        pcdv = o3d.geometry.PointCloud()
        pcdv.points = o3d.utility.Vector3dVector(vis_points)
        o3d.io.write_point_cloud(vis_path, pcdv)


# endregion


def manual_app():
    '''
    手动矫正代码
    :return:
    '''
    # region single box visualization
    radius = 5
    img_id = 131
    bbox = [[3960, 1100], [4900, 2250]]

    panorama_path = r'E:\demo\demo0617\img_insta\%d.jpg'%img_id
    densemap_path = r'E:\demo\demo0617\ColorizedPointCloud.pcd'
    trajectory_path = r'E:\demo\demo0617\img_insta\insta_2024_06_17_10_29_38.pcd'

    box_vis_path = r'E:\demo\demo0617\box_vis\%d.jpg'%img_id
    pcd_local_path = r'E:\demo\demo0617\temp\local.pcd'
    pan2pcd_path =r'E:\demo\demo0617\pan2pcd\%d.npy'%img_id
    pan2pcd_vis_path =r'E:\demo\demo0617\pan2pcd_vis\%d.jpg'%img_id

    pcdbox_vis_path =(r'E:\demo\demo0617\pcdbox\%d_%d_%d_%d_%d.pcd'%
                      (img_id, bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]))

    # img_show(panorama_path)
    # img_box_vis(panorama_path, bbox, box_vis_path)


    pan_img = io.imread(panorama_path)
    H,W = pan_img.shape[:2]
    print('H, W: ', H,W)

    center_pos, rot = get_posbytrajectory(panorama_path, trajectory_path)

    local_pcd = get_pcd(
        densemap_path,
        center_pos,
        radius=radius,
        save_path=pcd_local_path
    )

    map_pcd2pan(
        local_pcd,
        center_pos,
        H, W,
        director=rot,
        save_path=pan2pcd_path,
        vis_path=pan2pcd_vis_path,
        thicken=True
    )
    # box_pan2pcd(bbox, pan2pcd_path, vis_path=pcdbox_vis_path)
    box_pan2pcd_measure(bbox, pan2pcd_path, vis_path=pcdbox_vis_path)

    img_box_vis(pan2pcd_vis_path, bbox, box_vis_path)
    # endregion

def yolo_app():
    '''
    读取yolo预测结果，自动转换
    :return:
    '''

    panorama_dir = r'E:\demo\demo0617\img_insta'
    prediction_dir = r'E:\demo\demo0617\prediction'

    densemap_path = r'E:\demo\demo0617\ColorizedPointCloud.pcd'
    trajectory_path = r'E:\demo\demo0617\img_insta\insta_2024_06_17_10_29_38.pcd'
    pcd_local_path = r'E:\demo\demo0617\temp\local.pcd'

    img_box_vis_dir = r'E:\demo\demo0617\img_box_vis'
    pcdmap_box_vis_dir = r'E:\demo\demo0617\pcdmap_box_vis'
    pan2pcd_dir = r'E:\demo\demo0617\pan2pcd'
    pan2pcd_vis_dir = r'E:\demo\demo0617\pan2pcd_vis'
    pcd_box_dir = r'E:\demo\demo0617\pcd_box'
    H,W = 2880, 5760
    Radius = 5

    os.makedirs(img_box_vis_dir, exist_ok=True)
    os.makedirs(pcdmap_box_vis_dir, exist_ok=True)
    os.makedirs(pan2pcd_dir, exist_ok=True)
    os.makedirs(pan2pcd_vis_dir, exist_ok=True)
    os.makedirs(pcd_box_dir, exist_ok=True)

    img_list = [img_name for img_name in os.listdir(panorama_dir) if img_name.endswith('.jpg')]
    for img_name in tqdm(img_list):
        img_id = Path(img_name).stem
        panorama_path = os.path.join(panorama_dir, img_name)
        pred_path = os.path.join(prediction_dir, '%s.txt'%img_id)
        img_box_vis_path = os.path.join(img_box_vis_dir, '%s.jpg'%img_id)
        pcdmap_box_vis_path = os.path.join(pcdmap_box_vis_dir, '%s.jpg'%img_id)
        pan2pcd_path = os.path.join(pan2pcd_dir, '%s.npy'%img_id)
        pan2pcd_vis_path = os.path.join(pan2pcd_vis_dir, '%s.jpg'%img_id)
        pcd_box_path = os.path.join(pcd_box_dir, '%s.pcd'%img_id)

        try:
            df = pd.read_csv(pred_path, header=None, index_col=None, names=['class', 'x', 'y', 'w', 'h'], sep=' ')
        except Exception as e:
            print(pred_path, e)
            continue

        df['x_t'] = df['x'] * W
        df['y_t'] = df['y'] * H
        df['w_t'] = df['w'] * W
        df['h_t'] = df['h'] * H

        df['top_left_x'] = df['x_t'] - df['w_t'] / 2
        df['top_left_y'] = df['y_t'] - df['h_t'] / 2
        df['bottom_right_x'] = df['x_t'] + df['w_t'] / 2
        df['bottom_right_y'] = df['y_t'] + df['h_t'] / 2

        center_pos, rot = get_posbytrajectory(panorama_path, trajectory_path)

        local_pcd = get_pcd(
            densemap_path,
            center_pos,
            radius=Radius,
            save_path=pcd_local_path
        )

        map_pcd2pan(
            local_pcd,
            center_pos,
            H, W,
            director=rot,
            save_path=pan2pcd_path,
            vis_path=pan2pcd_vis_path,
            thicken=True
        )

        bboxes=[]
        for idx, row in df.iterrows():
            bbox = [
                [int(row['top_left_x']), int(row['top_left_y'])],
                [int(row['bottom_right_x']), int(row['bottom_right_y'])]
                    ]
            bboxes.append(bbox)
            box_pan2pcd_measure(bbox, pan2pcd_path, vis_path=pcd_box_path)

        img_boxes_vis(pan2pcd_vis_path, bboxes, pcdmap_box_vis_path)
        img_boxes_vis(panorama_path, bboxes, img_box_vis_path)







if __name__ == '__main__':
    pass
    yolo_app()
