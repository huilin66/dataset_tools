import os
from tqdm import tqdm

def attribute_update(input_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    labels_list = os.listdir(input_dir)
    for label_name in tqdm(labels_list):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(output_dir, label_name)

        with open(input_path, 'r') as f:
            lines = f.readlines()  # 读取所有行
            for idx, line in enumerate(lines):
                if line[8] == '1':
                    lines[idx] = lines[idx][:8]+'2'+lines[idx][9:]
        with open(output_path, 'w') as f:
            f.writelines(lines)

def class_update(input_dir, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    labels_list = os.listdir(input_dir)
    for label_name in tqdm(labels_list):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(output_dir, label_name)

        with open(input_path, 'r') as f:
            lines = f.readlines()  # 读取所有行
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_path, 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    pass
    root_dir = r'E:\data\202502_signboard\annotation_result_merge'
    labels_dir = os.path.join(root_dir, 'labels')
    labels_update_dir = os.path.join(root_dir, 'labels_update')
    # attribute_update(labels_dir, labels_update_dir)
    class_update(labels_dir, labels_update_dir)