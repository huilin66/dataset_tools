import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from tqdm import tqdm
from YOLO_slicing.slicing import slice_image
from YOLO_slicing.slicing import load_image_pil, calculate_slices_xyxy

from shapely import geometry

def load_label_seg(label):
    pass
    import pandas as pd
    # df = pd.read_csv(label, sep=' ', header=None, index_col=None)
    data = []
    with open(label, "r") as file:
        for line in file:
            row = list(map(float, line.strip().split(" ")))
            data.append(row)
    df = pd.DataFrame(data)
    label_array = df.to_numpy()
    return label_array

def array_xywh_xyxy_seg(label_array, img_w, img_h):
    scale_array = [1]
    repeat_num = int((label_array.shape[1]-1)/2)
    scale_array += [img_w, img_h]*repeat_num
    scale_array = np.array(scale_array)[np.newaxis, :]
    label_array[np.isnan(label_array)] = -1
    label_array *= scale_array
    label_array = np.array(label_array, dtype=np.int16)
    return label_array

def mask2box(mask):
    mask = np.array([mask[::2], mask[1::2]])
    x_min,y_min = np.min(mask, axis=1)
    x_max,y_max = np.max(mask, axis=1)
    box = [x_min, y_min, x_max, y_max]
    return box


def anotation_inside_slice_seg(mask,slice_coord):
    bbox = mask2box(mask)

    if slice_coord[0] >= bbox[2]:
        return False

    elif slice_coord[2] <= bbox[0]:
        return False

    elif slice_coord[1] >= bbox[3]:
        return False

    elif slice_coord[3] <= bbox[1]:
        return False

    else:
        return True

def intersect_xyxy_seg(mask, slice_coord):
    """
    Calculate intersection between two boxes

    Returns:
        List[int]: [xmin,ymin,xmax,ymax]
    """
    # 定义多边形的顶点坐标
    polygon_coords = np.array(mask).reshape((-1,2))
    xmin,ymin,xmax,ymax = slice_coord

    # 创建多边形对象和矩形对象
    polygon = geometry.Polygon(polygon_coords)
    rectangle = geometry.box(*slice_coord)  # 通过矩形的对角线坐标创建矩形对象
    # rectangle = geometry.Polygon([(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)])

    # 计算重叠区域
    overlap_area = polygon.intersection(rectangle)
    # print(overlap_area)



    # x, y = rectangle.exterior.xy
    # plt.plot(x, y, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=1)
    #
    # x, y = polygon.exterior.xy
    # plt.plot(x, y, color='blue', alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)
    #
    # # x, y = overlap_area.exterior.xy
    # # plt.plot(x, y, color='green', alpha=0.7, linewidth=2, solid_capstyle='round', zorder=3)
    #
    #
    # plt.show()
    if isinstance(overlap_area, shapely.Polygon):
        xs,ys = overlap_area.exterior.xy
        merged_list = [val for pair in zip(xs,ys) for val in pair]
        return merged_list, False
    elif isinstance(overlap_area, shapely.MultiPolygon):
        overlap_area_list = [polygon for polygon in overlap_area.geoms]
        merged_list_list = []
        for overlap_area in overlap_area_list:
            xs,ys = overlap_area.exterior.xy
            merged_list = [val for pair in zip(xs,ys) for val in pair]
            merged_list_list.append(merged_list)
        return merged_list_list, True

def rel_coord_xyxy_seg(mask, slice):
    patch_w = slice[2] - slice[0]
    patch_h = slice[3] - slice[1]
    repeat_num = int(len(mask)/2)

    mask = np.array(mask)

    move_array = repeat_num*[slice[0], slice[1]]
    scale_array = repeat_num*[patch_w, patch_h]

    mask -= np.array(move_array)

    scale_mask = mask/np.array(scale_array)
    scale_mask = scale_mask.astype(str).tolist()
    return scale_mask

def slice_image_seg(image, label, out_dir, slice_w, slice_h, overlap_w, overlap_h):
    """
    Slice image into cuts, save crops and labels

    Args:
    image(str): file path to the image
    label(str): file path to the label txt
    out_dir(str): output folder path
    slice_h(int): slice height
    slice_w(int): slice width
    overlap_h(float): overlap height ratio
    overlap_w(float): overlap width ratio

    Return:

    """

    img = load_image_pil(image)

    img_w, img_h = img.size

    # get list of slices coordinates
    slice_coords = calculate_slices_xyxy(img_h, img_w, slice_h, slice_w, overlap_h, overlap_w)
    # print(f'número de cortes por imagen: {len(slice_coords)}')

    # load labels as numpy array
    label_array = load_label_seg(label)

    # print(f'raw: {label_array}\nxyxy: {label_array_xyxy}')

    # main loop, check if label is inside the slice, if it is,calculate intersection and add row to txt label
    if label_array is not None:
        label_array_xyxy = array_xywh_xyxy_seg(label_array, img_w, img_h)
        for slice in slice_coords:

            # filename of the slice is like  image_filename_{xmin}_{ymin}_{xmax}_{ymax}
            slice_filename = (os.path.basename(image).replace('.png', '').replace('.jpg', '') +
                              f'_{slice[0]}_{slice[1]}_{slice[2]}_{slice[3]}')

            # save slice:
            img_slice = img.crop((slice[0], slice[1], slice[2], slice[3]))
            img_slice.save(os.path.join(out_dir, slice_filename) + '.jpg', quality=100)

            # path to save the txt file
            txt_path = os.path.join(out_dir, slice_filename + '.txt')

            # make txt file, after we can delete empty txt
            with open(txt_path, 'w') as f:

                for label in label_array_xyxy:
                    label = label[label>=0]
                    if label is None:
                        continue

                    # check if the label is inside the box
                    if anotation_inside_slice_seg(label[1:], slice) == True:
                        # print(f'********\nlabel: {label[1:]}\nslice: {slice}')   #TODO: borrar esto

                        # calculate intersection
                        intersections, mulit_result = intersect_xyxy_seg(label[1:], slice)
                        if not mulit_result:
                            rel_mask = rel_coord_xyxy_seg(intersections, slice)
                            write_str = ' '.join([str(label[0])]+rel_mask)
                            f.write(write_str+'\n')
                        else:
                            for intersection in intersections:
                                rel_mask = rel_coord_xyxy_seg(intersection, slice)
                                write_str = ' '.join([str(label[0])] + rel_mask)
                                f.write(write_str + '\n')
                f.close()


def check_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def yolo_slice(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir, tp_dir,
               slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5, seg=False):
    check_dir(output_img_dir)
    check_dir(output_txt_dir)
    check_dir(tp_dir)
    shutil.rmtree(tp_dir)
    check_dir(tp_dir)
    img_list = os.listdir(input_img_dir)
    suffix = '.' + img_list[0].split('.')[-1]
    # suffix = '.jpg'
    for img_name in tqdm(img_list):
        txt_name = img_name.replace(suffix, '.txt')
        input_image_path = os.path.join(input_img_dir, img_name)
        input_txt_path = os.path.join(input_txt_dir, txt_name)
        if seg:
            slice_image_seg(input_image_path, input_txt_path, tp_dir, slice_w, slice_h, overlap_w, overlap_h)
        else:
            slice_image(input_image_path, input_txt_path, tp_dir, slice_w, slice_h, overlap_w, overlap_h)

    img_list = [file_name for file_name in os.listdir(tp_dir) if file_name.endswith(suffix)]
    for img_name in tqdm(img_list):
        txt_name = img_name.replace(suffix, '.txt')
        tp_image_path = os.path.join(tp_dir, img_name)
        tp_txt_path = os.path.join(tp_dir, txt_name)
        output_image_path = os.path.join(output_img_dir, img_name)
        output_txt_path = os.path.join(output_txt_dir, txt_name)

        shutil.move(tp_image_path, output_image_path)
        shutil.move(tp_txt_path, output_txt_path)

from skimage import io
from pathlib import Path
def calculate_slices_xyxy(image_h,image_w,slice_h,slice_w,overlap_h,overlap_w):
    """
    Calculates the coordinates of the slices

    Args:
        image_h(int): image height
        image_w(int): image width
        slice_h(int): slice height
        slice_w(int): slice width
        overlap_h(float): overlap height ratio
        overlap_w(float): overlap width ratio

    Returns:
    List[List[int]]: list of slices coordinates [[xmin1,ymin1,xmax1,ymax1],...,[xminn,yminn,xmaxn,ymaxn]]
    """
    coords = []

    y_overlap = int(overlap_h * slice_h)
    x_overlap = int(overlap_w * slice_w)

    y_min=0
    y_max=0
    while y_max < image_h:
            x_min = x_max = 0
            y_max = y_min + slice_h
            while x_max < image_w:
                x_max = x_min + slice_w
                if y_max > image_h or x_max > image_w:
                    xmax = min(image_w, x_max)
                    ymax = min(image_h, y_max)
                    xmin = max(0, xmax - slice_w)
                    ymin = max(0, ymax - slice_h)
                    coords.append([xmin, ymin, xmax, ymax])
                else:
                    coords.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

    return coords

# region merge without category
def is_overlapping(box1, box2):
    # box 格式：[x_min, y_min, x_max, y_max]
    return not (box1[2] < box2[0] or  # box1 在 box2 左边
                box1[0] > box2[2] or  # box1 在 box2 右边
                box1[3] < box2[1] or  # box1 在 box2 上边
                box1[1] > box2[3])    # box1 在 box2 下边
def merge_two_boxes(box1, box2):
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return [x_min, y_min, x_max, y_max]
def group_boxes(boxes):
    n = len(boxes)
    visited = [False] * n
    groups = []

    def dfs(current, group):
        visited[current] = True
        group.append(boxes[current])
        for i in range(n):
            if not visited[i] and is_overlapping(boxes[current], boxes[i]):
                dfs(i, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)
            groups.append(group)

    return groups
def merge_all_overlapping_boxes(boxes):
    groups = group_boxes(boxes)
    merged_boxes = []

    for group in groups:
        x_min = min(box[0] for box in group)
        y_min = min(box[1] for box in group)
        x_max = max(box[2] for box in group)
        y_max = max(box[3] for box in group)
        merged_boxes.append([x_min, y_min, x_max, y_max])

    return merged_boxes
# endregion

# region merge with category
def is_overlapping_with_category(box1, box2):
    # box 格式：[cat_id, x_min, y_min, x_max, y_max]
    if box1[0] != box2[0]:  # 如果类别不同，不合并
        return False
    return not (box1[3] < box2[1] or  # box1 在 box2 上边
                box1[1] > box2[3] or  # box1 在 box2 下边
                box1[4] < box2[2] or  # box1 在 box2 左边
                box1[2] > box2[4])    # box1 在 box2 右边
def merge_two_boxes_with_category(box1, box2):
    # 合并框：类别不变，取边界的最小值和最大值
    cat_id = box1[0]
    x_min = min(box1[1], box2[1])
    y_min = min(box1[2], box2[2])
    x_max = max(box1[3], box2[3])
    y_max = max(box1[4], box2[4])
    return [cat_id, x_min, y_min, x_max, y_max]
def merge_all_overlapping_boxes_with_category(boxes):
    n = len(boxes)
    visited = [False] * n
    merged_boxes = []

    def dfs(current, group):
        visited[current] = True
        group.append(boxes[current])
        for i in range(n):
            if not visited[i] and is_overlapping_with_category(boxes[current], boxes[i]):
                dfs(i, group)

    for i in range(n):
        if not visited[i]:
            group = []
            dfs(i, group)

            # 合并组内所有框
            cat_id = group[0][0]  # 类别一致
            x_min = min(box[1] for box in group)
            y_min = min(box[2] for box in group)
            x_max = max(box[3] for box in group)
            y_max = max(box[4] for box in group)

            merged_boxes.append([cat_id, x_min, y_min, x_max, y_max])

    return merged_boxes
# endregion

def yolo_merge(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir, tp_dir,
               src_w=1920, src_h=1920, slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5, seg=False):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(tp_dir, exist_ok=True)

    img_list = os.listdir(input_img_dir)
    df = pd.DataFrame(img_list, columns=['img_name'])

    df['stem'] = df['img_name'].apply(lambda x: Path(x).stem)
    df['suffix'] = df['img_name'].apply(lambda x: Path(x).suffix)
    split_columns = ['src_name', 'name_x1', 'name_y1', 'name_x2', 'name_y2', ]
    df[split_columns] = df['stem'].str.split('_', expand=True, n=5)

    coords = calculate_slices_xyxy(src_h, src_w, slice_h, slice_w, overlap_h, overlap_w)
    img_src_list = df['src_name'].unique()
    suffix_list = df['suffix'].unique()
    assert len(suffix_list) == 1, print('multiple suffix', suffix_list)
    for src_name in tqdm(img_src_list):
        img_merge = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        # label_merge = pd.DataFrame(None, columns=['cat_id', 'x', 'y', 'w', 'h'])
        img_merge_name = src_name + suffix_list[0]
        label_merge_name = src_name + '.txt'
        img_merge_path = os.path.join(output_img_dir, img_merge_name)
        label_merge_path = os.path.join(output_txt_dir, label_merge_name)

        df_label_list = []
        for coord in coords:
            # image merge
            img_slice_name = src_name + f'_{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}' + suffix_list[0]
            img_slice_path = os.path.join(input_img_dir, img_slice_name)
            assert os.path.exists(img_slice_path), print(img_slice_name, 'does not exist!')
            img_slice = io.imread(img_slice_path)
            img_merge[coord[1]:coord[3],coord[0]:coord[2]] = img_slice

            # TODO: label merge for complex labels
            label_slice_name = src_name + f'_{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}' + '.txt'
            label_slice_path = os.path.join(input_txt_dir, label_slice_name)
            assert os.path.exists(label_slice_path), print(label_slice_name, 'does not exist!')
            df_label_slice = pd.read_csv(label_slice_path, header=None, index_col=None, names=['cat_id', 'x_rel_s', 'y_rel_s', 'w_rel_s', 'h_rel_s'], sep=' ')

            # 在slice小图上的位置
            df_label_slice['x1_rel_s'] = df_label_slice['x_rel_s'] - df_label_slice['w_rel_s'] * 0.5
            df_label_slice['y1_rel_s'] = df_label_slice['y_rel_s'] - df_label_slice['h_rel_s'] * 0.5
            df_label_slice['x2_rel_s'] = df_label_slice['x_rel_s'] + df_label_slice['w_rel_s'] * 0.5
            df_label_slice['y2_rel_s'] = df_label_slice['y_rel_s'] + df_label_slice['h_rel_s'] * 0.5

            df_label_slice['x1_s'] = df_label_slice['x1_rel_s'] * slice_w
            df_label_slice['y1_s'] = df_label_slice['y1_rel_s'] * slice_h
            df_label_slice['w_s'] = df_label_slice['w_rel_s'] * slice_w
            df_label_slice['h_s'] = df_label_slice['h_rel_s'] * slice_h
            df_label_slice['x2_s'] = df_label_slice['x1_s'] + df_label_slice['w_s']
            df_label_slice['y2_s'] = df_label_slice['y1_s'] + df_label_slice['h_s']

            # 在merge大图上的位置
            df_label_slice['x1_l'] = df_label_slice['x1_s'] + coord[0]
            df_label_slice['y1_l'] = df_label_slice['y1_s'] + coord[1]
            df_label_slice['x2_l'] = df_label_slice['x2_s'] + coord[0]
            df_label_slice['y2_l'] = df_label_slice['y2_s'] + coord[1]

            df_label_slice['x1_rel_l'] = df_label_slice['x1_l'] / src_w
            df_label_slice['y1_rel_l'] = df_label_slice['y1_l'] / src_h
            df_label_slice['x2_rel_l'] = df_label_slice['x2_l'] / src_w
            df_label_slice['y2_rel_l'] = df_label_slice['y2_l'] / src_h

            df_label_list.append(df_label_slice)

        df_label_list = pd.concat(df_label_list, axis=0)
        df_label_list['x_rel_l'] = (df_label_list['x1_rel_l'] + df_label_list['x2_rel_l']) / 2
        df_label_list['y_rel_l'] = (df_label_list['y1_rel_l'] + df_label_list['y2_rel_l']) / 2
        df_label_list['w_rel_l'] = df_label_list['x2_rel_l'] - df_label_list['x1_rel_l']
        df_label_list['h_rel_l'] = df_label_list['y2_rel_l'] - df_label_list['y1_rel_l']
        df_label_merge = df_label_list[['cat_id', 'x1_rel_l', 'y1_rel_l', 'x2_rel_l', 'y2_rel_l']]
        box_list = df_label_merge.values.tolist()
        merged_boxes = merge_all_overlapping_boxes_with_category(box_list)
        df_label_merge = pd.DataFrame(merged_boxes, columns= ['cat_id', 'x1_rel_l', 'y1_rel_l', 'x2_rel_l', 'y2_rel_l'])
        df_label_merge['x_rel_l'] = (df_label_merge['x1_rel_l'] + df_label_merge['x2_rel_l']) / 2
        df_label_merge['y_rel_l'] = (df_label_merge['y1_rel_l'] + df_label_merge['y2_rel_l']) / 2
        df_label_merge['w_rel_l'] = df_label_merge['x2_rel_l'] - df_label_merge['x1_rel_l']
        df_label_merge['h_rel_l'] = df_label_merge['y2_rel_l'] - df_label_merge['y1_rel_l']
        df_label_merge = df_label_merge[['cat_id', 'x_rel_l', 'y_rel_l', 'w_rel_l', 'h_rel_l']]
        df_label_merge.to_csv(label_merge_path, sep=' ', index=False, header=False)
        io.imsave(img_merge_path, img_merge)


def get_dataset(input_csv, output_csv):
    def replace_filename(row):
        idx = row.name % 3
        # base_filename = (row['filename'].replace('.jpg', '_%d_%d_%d_%d.jpg'%(idx*960, 0, (idx+2)*960, 1920)).
        #                  replace(os.path.dirname(input_csv), os.path.dirname(output_csv)))
        base_filename = (row['filename'].replace('.png', '_%d_%d_%d_%d.jpg'%(idx*960, 0, (idx+2)*960, 1920)).
                         replace(os.path.dirname(input_csv), os.path.dirname(output_csv)))
        return base_filename
    df = pd.read_csv(input_csv, header=None, index_col=None, names=['filename'])

    new_df = pd.concat([df] * 3, ignore_index=True)
    new_df['filename'] = new_df.apply(replace_filename, axis=1)

    new_df.to_csv(output_csv, header=None, index=None)


if __name__ == '__main__':
    pass

    # yolo_slice(input_img_dir=r'E:\data\0318_fireservice\data0327\images',
    #            input_txt_dir=r'E:\data\0318_fireservice\data0327\labels',
    #            output_img_dir=r'E:\data\0318_fireservice\data0327\images_slice',
    #            output_txt_dir=r'E:\data\0318_fireservice\data0327\labels_slice',
    #            tp_dir=r'E:\data\0318_fireservice\data0327\tp',
    #            slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5)

    # get_dataset(input_csv=r'E:\data\0318_fireservice\data0327\train.txt',
    #             output_csv=r'E:\data\0318_fireservice\data0327slice\train.txt',)
    # get_dataset(input_csv=r'E:\data\0318_fireservice\data0327\val.txt',
    #             output_csv=r'E:\data\0318_fireservice\data0327slice\val.txt',)


    # yolo_slice(input_img_dir=r'E:\data\0417_signboard\data0417\yolo\images',
    #            input_txt_dir=r'E:\data\0417_signboard\data0417\yolo\labels',
    #            output_img_dir=r'E:\data\0417_signboard\data0417\yolo\images_slice',
    #            output_txt_dir=r'E:\data\0417_signboard\data0417\yolo\labels_slice',
    #            tp_dir=r'E:\data\0417_signboard\data0417\tp',
    #            slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5)

    # yolo_slice(input_img_dir=r'E:\data\0417_signboard\data0417\yolo\images',
    #            input_txt_dir=r'E:\data\0417_signboard\data0417\yolo\labels',
    #            output_img_dir=r'E:\data\0417_signboard\data0417\yolo\images_slice',
    #            output_txt_dir=r'E:\data\0417_signboard\data0417\yolo\labels_slice',
    #            tp_dir=r'E:\data\0417_signboard\data0417\tp',
    #            slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5, seg=True)

    # get_dataset(input_csv=r'E:\data\0417_signboard\data0417\yolo\train.txt',
    #             output_csv=r'E:\data\0417_signboard\data0417\yolo_slice\train.txt',)
    # get_dataset(input_csv=r'E:\data\0417_signboard\data0417\yolo\val.txt',
    #             output_csv=r'E:\data\0417_signboard\data0417\yolo_slice\val.txt',)


    # 2880*5760 -> 1440*1440
    # yolo_slice(input_img_dir=r'E:\demo\demo_slice_merge\yolo\images',
    #            input_txt_dir=r'E:\demo\demo_slice_merge\yolo\labels',
    #            output_img_dir=r'E:\demo\demo_slice_merge\yolo\images_slice',
    #            output_txt_dir=r'E:\demo\demo_slice_merge\yolo\labels_slice',
    #            tp_dir=r'E:\demo\demo_slice_merge\yolo\tp',
    #            slice_w=1440, slice_h=1440, overlap_w=0.5, overlap_h=0.5, seg=False)
    # 1440*1440 -> 2280*5760
    yolo_merge(input_img_dir=r'E:\demo\demo_slice_merge\yolo\images_slice',
               input_txt_dir=r'E:\demo\demo_slice_merge\yolo\labels_slice',
               output_img_dir=r'E:\demo\demo_slice_merge\yolo\images_merge',
               output_txt_dir=r'E:\demo\demo_slice_merge\yolo\labels_merge',
               tp_dir=r'E:\data\0417_signboard\data0417\tp',
               src_h=2880, src_w=5760,
               slice_w=1440, slice_h=1440, overlap_w=0.5, overlap_h=0.5, seg=True)