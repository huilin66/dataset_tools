import os
import pandas as pd
from tqdm import tqdm


att_merge_dict = {
    0: 0,
    1: 1,
    2: 1,
    3: 3,
    4: 4,
    5: 4,
    6: 6,
    7: 7,
    8: 7,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
}
att_remove_list = [2, 5, 8, 3]

def yolo_att_merge(input_dir, output_dir, merge_dict, remove_list):
    os.makedirs(output_dir, exist_ok=True)
    gap_num = 2
    remove_list = [num+gap_num for num in remove_list]
    input_list = os.listdir(input_dir)
    for input_name in tqdm(input_list):
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, input_name)
        df = pd.read_csv(input_path, header=None, index_col=None, sep=' ')
        for k,v in merge_dict.items():
            if k==v:
                continue
            else:
                df[v+gap_num] = df[v+gap_num] | df[k+gap_num]

        df = df.drop(columns=remove_list)
        df[1] = len(merge_dict)-len(remove_list)
        df.to_csv(output_path, header=False, index=False, sep=' ')


if __name__ == '__main__':
    pass
    input_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5\labels'
    output_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection5_10\labels'
    yolo_att_merge(input_dir, output_dir, att_merge_dict, att_remove_list)