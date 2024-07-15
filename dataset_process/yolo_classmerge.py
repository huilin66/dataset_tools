import os
import pandas as pd
from tqdm import tqdm
import os.path as osp
# id_merge_dict = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 1,
#     5: 2,
#     6: 4,
# }

# id_merge_dict = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 2,
#     8: 1,
#     9: 2,
# }


# id_merge_dict = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
#     8: 8,
#     9: 9,
#     10: 9,
# }
# name_merge_dict = {
#     'Automatic Sprinkler': 'Automatic Sprinkler',
#     'Fire Detectors': 'Fire Detectors',
#     'Fire Alarm Bell': 'Fire Alarm Bell',
#     'Fire Alarm_break glass': 'Fire Alarm_break glass',
#     'Fire Hose Reel': 'Fire Hose Reel',
#     'Exit': 'Exit',
#     'Fire Alarm Bell round': 'Fire Alarm Bell round',
#     'Fire Alarm Bell white': 'Fire Alarm Bell',
#     'Fire Detectors white': 'Fire Detectors',
#     'Fire Alarm Bell flat': 'Fire Alarm Bell',
# }

class_merge_dict = {
    0: 0,
    1: 1,
    2: 1,
}

def yolo_idmerge(gt_dir, merge_dict):
    gt_list = os.listdir(gt_dir)
    for gt_name in tqdm(gt_list):
        gt_path = os.path.join(gt_dir, gt_name)
        df = pd.read_csv(gt_path, names=['catid', 'x1', 'x2', 'y1', 'y2'], header=None, index_col=None, sep=' ')
        for idx, row in df.iterrows():
            src_id = row['catid']
            dst_id = merge_dict[src_id]
            df.loc[idx, 'catid'] = dst_id
        df.to_csv(gt_path, header=None, index=None, sep=' ')

def yolo_classmerge(input_dir, copy_dir):
    '''

    :param input_dir:
    :param copy_dir:
    :return:
    '''
    os.makedirs(copy_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_file = osp.join(input_dir, file_name)
        copy_file = osp.join(copy_dir, file_name)

        with open(input_file, 'r') as file:
            lines = file.readlines()
        filtered_lines = []
        for line in lines:
            numbers_str = line.split()
            numbers = [int(float(num_str)) if '.' not in num_str else float(num_str) for num_str in numbers_str]
            numbers[0] = class_merge_dict[numbers[0]]
            numbers_str = [str(num) for num in numbers]
            new_line = ' '.join(numbers_str)+'\n'
            filtered_lines.append(new_line)
        with open(copy_file, 'w') as file:
            file.writelines(filtered_lines)

if __name__ == '__main__':
    pass
    # labels_dir = r'E:\data\0318_fireservice\data0327\labels7'
    # yolo_idmerge(labels_dir, id_merge_dict)
    # labels_dir = r'E:\data\0318_fireservice\data0327slice\labels7'
    # yolo_idmerge(labels_dir, id_merge_dict)
    # labels_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection2\labels'
    # yolo_idmerge(labels_dir, id_merge_dict)

    yolo_classmerge(input_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5\labels',
                    copy_dir=r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection6\labels')