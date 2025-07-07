import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def mseg2seg(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                attribute_num = int(num_list[1])
                num_list_seg = num_list[:1]+num_list[1+attribute_num+1:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)
def mseg_class_update(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)


def seg_class_update(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def seg_filter_small(input_dir, output_dir, threshold, class_list, with_attribute):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        shutil.copy(input_image_path, output_image_path)

        filter_yolo_segmentation(input_label_path, output_label_path, threshold=threshold,
                                 class_list=class_list, with_attribute=with_attribute)


def filter_yolo_segmentation(input_file, output_file, threshold, with_attribute=False, class_list=[]):
    def calculate_polygon_area(points):
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0
    def clalulate_bbox_area(points):
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        x_min, x_max = min(x_points), max(x_points)
        y_min, y_max = min(y_points), max(y_points)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        return area

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        src_count, dst_count = 0, 0
        for line in f_in:
            src_count += 1
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id not in class_list:
                f_out.write(line + '\n')
                dst_count += 1
                continue
            if with_attribute:
                att_len= int(parts[1])
                coords = list(map(float, parts[2+att_len:]))
            else:
                coords = list(map(float, parts[1:]))


            normalized_points = [(coords[i], coords[i + 1])
                                 for i in range(0, len(coords), 2)]

            area = clalulate_bbox_area(normalized_points)
            if area >= threshold:
                f_out.write(line + '\n')
                dst_count += 1
        if src_count != dst_count:
            print(f'{os.path.basename(input_file)} change from {src_count} --> {dst_count}')


def data_move(input_dir, output_dir, move_gap=4000):
    img_list = os.listdir(input_dir)
    output_list_list = [img_list[i:i+move_gap] for i in range(0, len(img_list), move_gap)]

    for idx, output_list in enumerate(output_list_list):
        output_dir_path = output_dir+'%d'%idx
        os.makedirs(output_dir_path, exist_ok=True)
        for file_name in tqdm(output_list, desc=f'{idx}/{len(output_list_list)}'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir_path, file_name)
            shutil.move(input_path, output_path)

def result2csv(result_dir, csv_path):
    file_list = os.listdir(result_dir)
    df = pd.DataFrame(None, columns=['file_name', 'object_id', 'class_id', 'conf',
                                     'att_deformation', 'att_broken', 'att_abandonment', 'att_corrosion'])
    for file_name in tqdm(file_list):
        result_path = os.path.join(result_dir, file_name)
        with open(result_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.split()
                class_id = int(parts[0])
                conf = float(parts[-1])
                att_deformation = int(parts[2])
                att_broken = int(parts[3])
                att_abandonment = int(parts[4])
                att_corrosion = int(parts[5])
                df.loc[len(df)] = [file_name, idx, class_id, conf, att_deformation, att_broken, att_abandonment, att_corrosion]
    df.to_csv(csv_path)
    print(f'save {len(file_list)} files with {len(df)} result')

def results2result(results_list, result_path):
    pass
    df_list = [pd.read_csv(csv_path, header=0, index_col=0) for csv_path in results_list]
    df = pd.concat(df_list, axis=0)
    df.to_csv(result_path)
    print(f'save {os.path.basename(result_path)} with {len(df)} records!')


def result_sta(csv_path, show=False):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    unique_files = df['file_name'].unique()
    if show:
        confs = df['conf'].tolist()
        confs_sort = np.sort(confs)
        y = np.arange(1, len(confs_sort)+1)
        plt.plot(confs_sort, y)
        plt.grid(True)
        plt.show()

    df_with_defect = df[(df['att_deformation']!=0) | (df['att_broken']!=0) |
                        (df['att_abandonment']!=0) | (df['att_corrosion']!=0)]
    df_with_highdefect = df[(df['att_deformation']==2) | (df['att_broken']==2) |
                        (df['att_abandonment']==2) | (df['att_corrosion']==2)]
    unique_files_with_defect = df_with_defect['file_name'].unique()
    unique_files_with_highdefect  = df_with_highdefect ['file_name'].unique()
    if show:
        confs_with_defect = df_with_defect['conf'].tolist()
        confs_with_defect_sort = np.sort(confs_with_defect)
        y_with_defect = np.arange(1, len(confs_with_defect_sort)+1)
        plt.plot(confs_with_defect_sort, y_with_defect)
        plt.grid(True)
        plt.show()

    print(f'all result num:{len(df)}, file num:{len(unique_files)}\n'
          f'all defected result num:{len(df_with_defect)}  file num:{len(unique_files_with_defect)}\n'
          f'all highly defected result num:{len(df_with_highdefect)}  file num:{len(unique_files_with_highdefect)}')

    CONF_THRESHOLDS = np.linspace(0, 1, 21)[15:-1]
    for conf_threshold in CONF_THRESHOLDS:
        df_conf = df[df['conf']>=conf_threshold]
        df_conf_with_defect = df_conf[(df_conf['att_deformation']!=0) | (df_conf['att_broken']!=0) |
                              (df_conf['att_abandonment']!=0) | (df_conf['att_corrosion']!=0)]
        df_conf_with_highdefect = df_conf[(df_conf['att_deformation'] == 2) | (df_conf['att_broken'] == 2) |
                                (df_conf['att_abandonment'] == 2) | (df_conf['att_corrosion'] == 2)]
        unique_files = df_conf['file_name'].unique()
        unique_files_with_defect = df_conf_with_defect['file_name'].unique()
        unique_files_with_highdefect = df_conf_with_highdefect['file_name'].unique()
        print(f'\n CONF_THRESHOLD: {conf_threshold}')
        print(f'all result num:{len(df_conf)}, file num:{len(unique_files)}\n'
              f'all defected result num:{len(df_conf_with_defect)}  file num:{len(unique_files_with_defect)}\n'
              f'all highly defected result num:{len(df_with_highdefect)}  file num:{len(unique_files_with_highdefect)}')


def result_move(input_dir_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, input_dir in enumerate(input_dir_list):
        file_list = os.listdir(input_dir)
        for file_name in tqdm(file_list, desc=f'{idx}/{len(input_dir_list)}'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            shutil.copy(input_path, output_path)

def get_pseudo_labelling(result_csv_path, input_img_dir, input_label_dir, output_dir):
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    df = pd.read_csv(result_csv_path, header=0, index_col=0)
    df_conf = df[df['conf'] >= 0.7]
    # df_conf_with_highdefect = df_conf[(df_conf['att_deformation'] == 2) | (df_conf['att_broken'] == 2) |
    #                                   (df_conf['att_abandonment'] == 2) | (df_conf['att_corrosion'] == 2)]
    df_conf_with_highdefect = df_conf[(df_conf['att_deformation'] != 0) | (df_conf['att_broken'] != 0) |
                                  (df_conf['att_abandonment'] != 0) | (df_conf['att_corrosion'] != 0)]
    unique_files_with_highdefect = df_conf_with_highdefect['file_name'].unique()

    img_list = os.listdir(input_img_dir)
    for image_name in tqdm(img_list):
        label_name = Path(image_name).stem + '.txt'
        if label_name in unique_files_with_highdefect:
            input_img_path = os.path.join(input_img_dir, image_name)
            output_img_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_img_path, output_img_path)

            input_label_path = os.path.join(input_label_dir, label_name)
            output_label_path = os.path.join(output_label_dir, label_name)
            with open(input_label_path, 'r', encoding='utf-8') as fr, open(output_label_path, 'w', encoding='utf-8') as fw:
                lines = fr.readlines()
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    parts = parts[:-1]
                    line_new = ' '.join(parts)
                    fw.write(line_new+'\n')

def copy_dataset(input_dir, output_dir):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        shutil.copy(input_image_path, output_image_path)
        shutil.copy(input_label_path, output_label_path)


if __name__ == '__main__':
    pass
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg', cp_img=True)
    # seg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_c6', cp_img=True)

    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001',
    #                  threshold=0.01, class_list=[2, 4, 5, 7], with_attribute=True)
    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005',
    #                  threshold=0.05, class_list=[2, 4, 5, 7], with_attribute=True)
    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010',
    #                  threshold=0.10, class_list=[2, 4, 5, 7], with_attribute=True)

    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6_seg', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6_seg', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6_seg', cp_img=True)


    # mseg_class_update(input_dir=r'/data/huilin/data/isds/bd_data/data389',
    #          output_dir=r'//data/huilin/data/isds/bd_data/data389_c6', cp_img=True)

    # data_move(r'/data/huilin/data/isds/bd_data/data_all', '/data/huilin/data/isds/bd_data/data_all_split')
    # result2csv(r'/data/huilin/projects/ultralytics/runs/msegment/predict/labels',
    #            r'/data/huilin/data/isds/bd_data/data_all_split0.csv')
    # result2csv(r'/data/huilin/projects/ultralytics/runs/msegment/predict2/labels',
    #            r'/data/huilin/data/isds/bd_data/data_all_split1.csv')
    # result2csv(r'/data/huilin/projects/ultralytics/runs/msegment/predict3/labels',
    #            r'/data/huilin/data/isds/bd_data/data_all_split2.csv')
    # result2csv(r'/data/huilin/projects/ultralytics/runs/msegment/predict4/labels',
    #            r'/data/huilin/data/isds/bd_data/data_all_split3.csv')
    # result2csv(r'/data/huilin/projects/ultralytics/runs/msegment/predict5/labels',
    #            r'/data/huilin/data/isds/bd_data/data_all_split4.csv')

    # results2result([r'/data/huilin/data/isds/bd_data/data_all_split0.csv',
    #                 r'/data/huilin/data/isds/bd_data/data_all_split1.csv',
    #                 r'/data/huilin/data/isds/bd_data/data_all_split2.csv',
    #                 r'/data/huilin/data/isds/bd_data/data_all_split3.csv',
    #                 r'/data/huilin/data/isds/bd_data/data_all_split4.csv'],
    #                 r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                )

    # result_sta(r'/data/huilin/data/isds/bd_data/data_all.csv')
    #
    # result_move([r'/data/huilin/projects/ultralytics/runs/msegment/predict/labels',
    #              r'/data/huilin/projects/ultralytics/runs/msegment/predict2/labels',
    #              r'/data/huilin/projects/ultralytics/runs/msegment/predict3/labels',
    #              r'/data/huilin/projects/ultralytics/runs/msegment/predict4/labels',
    #              r'/data/huilin/projects/ultralytics/runs/msegment/predict5/labels',],
    #             r'/data/huilin/data/isds/bd_data/data_all_predict',
    #             )

    # get_pseudo_labelling(r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                      r'/data/huilin/data/isds/bd_data/data_all_split0',
    #                      r'/data/huilin/projects/ultralytics/runs/msegment/predict/labels',
    #                      r'/data/huilin/data/isds/bd_data/pseudo_data',
    #                      )
    # get_pseudo_labelling(r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                      r'/data/huilin/data/isds/bd_data/data_all_split1',
    #                      r'/data/huilin/projects/ultralytics/runs/msegment/predict2/labels',
    #                      r'/data/huilin/data/isds/bd_data/pseudo_data',
    #                      )
    # get_pseudo_labelling(r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                      r'/data/huilin/data/isds/bd_data/data_all_split2',
    #                      r'/data/huilin/projects/ultralytics/runs/msegment/predict3/labels',
    #                      r'/data/huilin/data/isds/bd_data/pseudo_data',
    #                      )
    # get_pseudo_labelling(r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                      r'/data/huilin/data/isds/bd_data/data_all_split3',
    #                      r'/data/huilin/projects/ultralytics/runs/msegment/predict4/labels',
    #                      r'/data/huilin/data/isds/bd_data/pseudo_data',
    #                      )
    # get_pseudo_labelling(r'/data/huilin/data/isds/bd_data/data_all.csv',
    #                      r'/data/huilin/data/isds/bd_data/data_all_split4',
    #                      r'/data/huilin/projects/ultralytics/runs/msegment/predict5/labels',
    #                      r'/data/huilin/data/isds/bd_data/pseudo_data',
    #                      )
    # copy_dataset(r'/data/huilin/data/isds/bd_data/data389_c6', r'/data/huilin/data/isds/bd_data/pseudo_data')

    mseg2seg(input_dir=r'/data/huilin/data/isds/bd_data/data389_c6',
             output_dir=r'/data/huilin/data/isds/bd_data/data389_c6_seg', cp_img=True)
