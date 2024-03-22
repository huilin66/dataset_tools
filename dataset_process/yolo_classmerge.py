import os
from tqdm import tqdm
merge_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 1,
    5: 2,
    6: 4,
}

def yolo_cm(labels_dir, merge_dict):
    # 遍历目录下的所有文件
    for filename in tqdm(os.listdir(labels_dir)):
        if filename.endswith(".txt"):
            path = os.path.join(labels_dir, filename)
            with open(path, 'r') as file:
                lines = file.readlines()

            with open(path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()

                    parts[0] = merge_dict[parts[0]]  # 将类别标签变为新的类别0
                    file.write(' '.join(parts) + '\n')


if __name__ == '__main__':
    pass
    labels_dir = r''
    yolo_cm(labels_dir, merge_dict)
