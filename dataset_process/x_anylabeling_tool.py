import os
import json
from tqdm import tqdm

def remove_attribute(input_dir):
    file_list = os.listdir(input_dir)
    file_obs_count = 0
    for file_name in tqdm(file_list):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='UTF-8') as file:
            json_data = json.load(file)
        for idx in range(len(json_data['shapes'])):
            file_obs_flag = False
            if 'attributes' in json_data['shapes'][idx] and 'surface obstructed' in json_data['shapes'][idx]['attributes']:
                del json_data['shapes'][idx]['attributes']['surface obstructed']
                file_obs_flag = True
        if file_obs_flag:
            file_obs_count += 1
        with open(file_path, 'w') as file:
            json.dump(json_data, file, indent=2)
    print('process %d "surface obstructed" files in %d'%(file_obs_count, len(file_list)))


def revalue_attribute(input_dir):
    file_list = os.listdir(input_dir)
    file_inc_count, file_cor_count, file_pee_count = 0, 0, 0
    for file_name in tqdm(file_list):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='UTF-8') as file:
            json_data = json.load(file)
        for idx in range(len(json_data['shapes'])):
            file_inc_flag, file_cor_flag, file_pee_flag = False, False, False
            if 'attributes' in json_data['shapes'][idx] and 'surface incomplete' in json_data['shapes'][idx]['attributes'] and json_data['shapes'][idx]['attributes']['surface incomplete'] != 'no':
                json_data['shapes'][idx]['attributes']['surface incomplete'] = 'yes'
                file_inc_flag = True
            if 'attributes' in json_data['shapes'][idx] and 'surface corroded' in json_data['shapes'][idx]['attributes'] and json_data['shapes'][idx]['attributes']['surface corroded'] != 'no':
                json_data['shapes'][idx]['attributes']['surface corroded'] = 'yes'
                file_cor_flag = True
            if 'attributes' in json_data['shapes'][idx] and 'surface peeling' in json_data['shapes'][idx]['attributes'] and json_data['shapes'][idx]['attributes']['surface peeling'] != 'no':
                json_data['shapes'][idx]['attributes']['surface peeling'] = 'yes'
                file_pee_flag = True
        if file_inc_flag:
            file_inc_count += 1
        if file_cor_flag:
            file_cor_count += 1
        if file_pee_flag:
            file_pee_count += 1

        with open(file_path, 'w') as file:
            json.dump(json_data, file, indent=2)
    print('process '
          '%d "surface incomplete", '
          '%d "surface incomplete", '
          '%d "surface incomplete", '
          'files in %d'%(file_inc_count, file_cor_count, file_pee_count, len(file_list)))

if __name__ == '__main__':
    pass
    remove_attribute(r'E:\data\0417_signboard\data0806_m\src\rgb_detection5')
    revalue_attribute(r'E:\data\0417_signboard\data0806_m\src\rgb_detection5')