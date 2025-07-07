import os
from tqdm import tqdm
import pandas as pd

def result_check(input_dir, output_dir):
    input_list = os.listdir(input_dir)
    for input_file in tqdm(input_list):
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file.replace('.png', '.csv'))
        if os.path.exists(output_path):
            continue
        else:
            print(output_path)

def track_check(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_list = os.listdir(input_dir)
    for input_file in tqdm(input_list):
        if input_file.endswith('.txt'):
            df = pd.read_csv(os.path.join(input_dir, input_file), sep=',', header=None, index_col=None)
            if df[1].min() >1:
                print(input_file)
                df[1] -= df[1].min()
            df.to_csv(os.path.join(output_dir, input_file), index=False, header=False)
                # print(df)
            # break
if __name__ == '__main__':
    pass
    # result_check(
    #     input_dir=r'E:\data\tp\sar_det\images',
    #     output_dir=r'E:\data\tp\sar_det\labels'
    # )
    track_check(r'E:\data\tp\sar_det\TestA', r'E:\data\tp\sar_det\result')