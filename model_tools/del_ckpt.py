import os
from tqdm import tqdm

def rm_ckpt(input_dir):
    for file_name in os.listdir(input_dir):
        if file_name != 'best_model.pdparams':
            file_path = os.path.join(input_dir, file_name)
            # os.remove(file_path)
            print('remove', file_path)


def rm_ckpts(root_dir):
    for dir_name in tqdm(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            rm_ckpt(dir_path)


if __name__ == '__main__':
    pass
    root_path = r'E:\repository\PaddleDetection\output'
    rm_ckpts(root_path)