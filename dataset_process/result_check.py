import os
from tqdm import tqdm


def result_check(input_dir, output_dir):
    input_list = os.listdir(input_dir)
    for input_file in tqdm(input_list):
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, input_file.replace('.png', '.csv'))
        if os.path.exists(output_path):
            continue
        else:
            print(output_path)
if __name__ == '__main__':
    result_check(
        input_dir=r'E:\data\tp\sar_det\images',
        output_dir=r'E:\data\tp\sar_det\labels'
    )