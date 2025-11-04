import os

from tqdm import tqdm

def list_remove_index(input_list, remove_index):
    output_list = [item for i, item in enumerate(input_list) if i not in remove_index]
    return output_list

def extract_single_risk_keep_len(input_dir, output_dir, risk):
    save_dir = os.path.join(output_dir, risk, 'labels')
    os.makedirs(save_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list, desc=risk):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(save_dir, label_name)
        with open(input_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                if risk == 'd':
                    parts[3], parts[4], parts[5] = '0', '0', '0'
                elif risk == 'b':
                    parts[2], parts[4], parts[5] = '0', '0', '0'
                elif risk == 'a':
                    parts[2], parts[3], parts[5] = '0', '0', '0'
                elif risk == 'c':
                    parts[2], parts[3], parts[4] = '0', '0', '0'
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)


def extract_single_risk_keep_single(input_dir, output_dir, risk):
    save_dir = os.path.join(output_dir, risk, 'labels')
    os.makedirs(save_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    for label_name in tqdm(label_list, desc=risk):
        input_path = os.path.join(input_dir, label_name)
        output_path = os.path.join(save_dir, label_name)
        with open(input_path, 'r') as f:
            lines = f.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                parts[1] = '1'
                if risk == 'd':
                    parts = list_remove_index(parts, [3, 4, 5])
                elif risk == 'b':
                    parts = list_remove_index(parts, [2, 4, 5])
                elif risk == 'a':
                    parts = list_remove_index(parts, [2, 3, 5])
                elif risk == 'c':
                    parts = list_remove_index(parts, [2, 3, 4])
                else:
                    ValueError(f"{risk} risk must be 'd' or 'b' or 'a' or 'c'")
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

        with open(output_path, 'w') as f:
            f.writelines(new_lines)