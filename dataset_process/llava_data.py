import os
import json
from tqdm import tqdm
import pandas as pd

id2cat_map = {
    0: 'background',
    1: 'crack',
    2: 'concrete_spalling',
    3: 'finishes_peeling',
    4: 'mold',
    5: 'water_seepage',
}
cat2id_map = {
    'background':0,
    'crack':1,
    'concrete_spalling':2,
    'finishes_peeling':3,
    'mold':4,
    'water_seepage':5,
}
id2cause_map = {
    # 0: 'background',
    0: 'heavy traffic, temperature changes, and other factors.',
    1: 'shrink age or contraction of the rendering on the walls or poor workmanship during construction.',
    2: 'ageing buildings; prolonged seepage of water damages the steel bars inside the concrete; steel bars become rusty.',
    3: 'ageing, structural movements, poor workmanship during installation thermal movement, inadequate expansion joints damage by external factor, ingress of water into the gap between the finishes and the surface of the wall.',
    4: 'Moisture from wet areas, vandalism or accidents, impacts from occupants or loads, deteriorates faster than expected.',
    5: 'defective fabric or installations of buildings and the lack of proper maintenance, leakage from defective water pipes, sanitary fitments or drainage pipes.',

}

id2action_map = {
    # 0: 'background',
    0: 'Contact the highway department to repair the defect.',
    1: 'Building owners should arrange for timely repair and maintenance works to upkeep the building in good condition.',
    2: 'Building owners should arrange for timely repair and maintenance works to upkeep the building in good condition.',
    3: 'Good management of the building is the key to maintaining building safety. Building owners are therefore advised to appoint competent management companies to manage their buildings.',
    4: 'Building owners should keep in view the conditions of the defects,unless the circumstances have changed, owners may carry out repair as necessary.',
    5: 'owners should investigate the source of seepage by liaising with owners of the flat concerned for carrying out repair works as early as possible,',
}

def get_gt_info(gt_path):
    def get_level(df):
        if len(df) > 2:
            return 'serious'
        elif len(df) == 2:
            if df['area'].max() < 0.1 and df['w_box'].max() < 0.1 and df['h_box'].max() < 0.1:
                return 'moderate'
            else:
                return 'serious'
        else:
            if df['area'].max() < 0.1 and df['w_box'].max() < 0.1 and df['h_box'].max() < 0.1:
                return 'slight'
            else:
                return 'serious'
    gt_info = {}
    df = pd.read_csv(gt_path, header=None, index_col=None, sep=' ',
                     names=['cat_id', 'x_center', 'y_center', 'w_box', 'h_box'])
    df['area'] = df['w_box']*df['h_box']
    cat_list = []
    for idx,row in df.iterrows():
        cat_name = id2cat_map[row['cat_id']]
        cat_list.append(cat_name)
    cat_set = list(set(cat_list))
    gt_info['type'] = ';'.join(cat_set)
    gt_info['number'] = len(df)
    gt_info['level'] = get_level(df)
    causes,actions = [],[]
    for cat_name in cat_list:
        cat_id = cat2id_map[cat_name]
        cause = id2cause_map[cat_id]
        action = id2action_map[cat_id]
        causes.append(cause)
        actions.append(action)
    gt_info['cause'] = causes[0] #';'.join(causes)
    gt_info['action'] = actions[0] #';'.join(actions)
    return gt_info


def get_img_info(img_path, gt_path):
    gt_info = get_gt_info(gt_path)
    img_info = {}
    img_info['id'] = os.path.basename(img_path).replace('.jpg', '')
    img_info['image'] = 'defect/'+ os.path.basename(img_path)
    img_info['conversations'] = [
        {
            "from": "human",
            "value": "<image>\nplease describe this image in table format."
        },
        {
            "from": "gpt",
            "value": "| property | value |\n"
                     "| --- | --- |\n"
                     "| background | %s |\n"
                     "| defect types | %s |\n"
                     "| defect numbers | %s |\n"
                     "| defect level | %s |\n"
                     "| possible causes of the defects | %s |\n"
                     "| required actions of the defects| %s |"%
                     ('road', 'crack', gt_info['number'], gt_info['level'], gt_info['cause'], gt_info['action'])
        },
    ]
    if gt_info['type'] != 'background':
        print(gt_info['type'])
    return img_info


def det2llava(img_dir, gt_dir, dst_json):
    js_data = []
    img_list = [os.path.join(img_dir, file_name) for file_name in os.listdir(img_dir)]
    gt_list = [img_path.replace(img_dir, gt_dir).replace('.jpg', '.txt') for img_path in img_list]
    for idx in tqdm(range(len(img_list))):
        img_path, gt_path = img_list[idx], gt_list[idx]
        img_info = get_img_info(img_path, gt_path)
        js_data.append(img_info)
    with open(dst_json, 'w') as f:
        output_json = json.dumps(js_data)
        f.write(output_json)


if __name__ == '__main__':
    pass
    # img_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\images\train'
    # gt_dir = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\labels\train'
    # dst_json = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect2k.json'
    # det2llava(img_dir, gt_dir, dst_json)

    # img_dir = r'E:\data\2023_defect\road_crack_detection.v2i.yolov9\train\images'
    # gt_dir = r'E:\data\2023_defect\road_crack_detection.v2i.yolov9\train\labels'
    # dst_json = r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect1k.json'
    # det2llava(img_dir, gt_dir, dst_json)

    data1 = json.load(open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect1k.json'))
    data2 = json.load(open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect2k.json'))

    data = data1+data2
    print(len(data1), len(data2), len(data))
    with open(r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\defect4k.json', 'w') as f:
        output_json = json.dumps(data)
        f.write(output_json)