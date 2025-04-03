import os
import json
import pprint
import shutil
from pathlib import Path
from tqdm import tqdm


def is_img(file_path):
    path = Path(file_path)
    ext = path.suffix.lower()
    return ext in ['.jpg', '.jpeg', '.png']


def xanylabel_check(input_dir, attribute_file):
    file_list = os.listdir(input_dir)
    json_list = [file_path.endswith('.json') for file_path in file_list]

    attributes_dict = json.load(open(attribute_file))

    for json_file in tqdm(json_list):
        with open(json_file, 'r') as f:
            data = json.load(f)
            shapes = data['shapes']
            for i in range(len(shapes)):
                label = shapes[i]['label']
                attributes = shapes[i]['attributes']

def labels_info(input_dir):
    file_list = [os.path.join(input_dir, file_name) for file_name in os.listdir(input_dir) if file_name.endswith('.json')]
    labels = {}
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as f:
            records = json.load(f)['shapes']
        for record in records:
            category = record['label']
            attributes = record['attributes']
            if category not in labels:
                labels[category] = {}
            for k,v in attributes.items():
                if k not in labels[category]:
                    labels[category][k] = [v]
                elif v not in labels[category][k]:
                    labels[category][k].append(v)
                else:
                    continue
    pprint.pprint(labels, indent=4)


def labels_replace(input_dir, output_dir, cat_map_dict, risk_map_dict, level_map_dict):
    os.makedirs(output_dir, exist_ok=True)
    img_list = [file_name for file_name in os.listdir(input_dir) if is_img(file_name)]
    for img_name in tqdm(img_list):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        shutil.copyfile(input_path, output_path)

    json_list = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.json')]
    for file_name in tqdm(json_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        with open(input_path, 'r') as f:
            data = json.load(f)
        records = data['shapes']
        for idx, record in enumerate(records):
            records[idx]['label'] = cat_map_dict[record['label']]
            attributes = record['attributes']
            attributes_new = {}
            for k,v in attributes.items():
                dst_k = risk_map_dict[k]
                dst_v = level_map_dict[v]
                attributes_new[dst_k] = dst_v
            record['attributes'] = attributes_new

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    # input_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\merge_data'
    # attribute_file = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\attributes_v1.json'
    # xanylabel_check(input_dir, attribute_file)
    input_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\merge_data'
    output_dir = r'E:\data\202502_signboard\20250224 Signboard Data and CDU\Selected_Sample\merge_data_v4'
    cat_map_dict = {
        'background': 'background',
        'wall surface': 'wall display',
        'wall frame': 'wall frame',
        'projecting surface': 'projecting display',
        'projecting frame': 'projecting frame',
        'hung surface': 'hanging display',
        'hung frame': 'hanging frame',
    }
    risk_map_dict = {
        'abandonment': 'abandonment',
        'broken': 'broken',
        'corrosion': 'corrosion',
        'deformation': 'deformation',
    }
    level_map_dict = {
        'no': 'no',
        'medium': 'medium',
        'high': 'high',
        'low': 'medium'
    }
    labels_info(input_dir)
    labels_replace(input_dir, output_dir, cat_map_dict, risk_map_dict, level_map_dict)
    labels_info(output_dir)