import json
import os
import cv2
from tqdm import tqdm
import numpy as np

COLOR_PALETTE = [
    (255, 42, 4),
    (79, 68, 255),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (0, 192, 38),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (104, 31, 17),
]

def time2name(input_dir):
    img_list = os.listdir(input_dir)
    time_list = [img_name.split(".")[0].split("_")[1] for img_name in img_list]
    time2name = dict(zip(time_list, img_list))
    return time2name

def dir_vis(json_dir, input_dir1, input_dir2, input_dir3, input_dir4, input_dir5, input_dir6):
    input1_time2name = time2name(input_dir1)
    input2_time2name = time2name(input_dir2)
    input3_time2name = time2name(input_dir3)
    input4_time2name = time2name(input_dir4)
    input5_time2name = time2name(input_dir5)
    input6_time2name = time2name(input_dir6)
    json_list = os.listdir(json_dir)
    for json_name in tqdm(json_list):
        json_path = os.path.join(json_dir, json_name)
        image_name = json_name.replace('.json', '')
        input1_path = os.path.join(input_dir1, input1_time2name[image_name] if image_name in input1_time2name else 'None')
        input2_path = os.path.join(input_dir2, input2_time2name[image_name] if image_name in input2_time2name else 'None')
        input3_path = os.path.join(input_dir3, input3_time2name[image_name] if image_name in input3_time2name else 'None')
        input4_path = os.path.join(input_dir4, input4_time2name[image_name] if image_name in input4_time2name else 'None')
        input5_path = os.path.join(input_dir5, input5_time2name[image_name] if image_name in input5_time2name else 'None')
        input6_path = os.path.join(input_dir6, input6_time2name[image_name] if image_name in input6_time2name else 'None')
        if os.path.exists(input1_path) and os.path.exists(input2_path) and os.path.exists(input3_path) and os.path.exists(input4_path) and os.path.exists(input5_path) and os.path.exists(input6_path) and os.path.exists(json_path):   
            files_vis(json_path, input1_path, input2_path, input3_path, input4_path, input5_path, input6_path)
        else:
            print(f'{json_name} sample missing')


def files_vis(json_path, input1_path, input2_path, input3_path, input4_path, input5_path, input6_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data1 = data[0]
    data2 = data[1]
    data3 = data[2]
    data4 = data[3]
    data5 = data[4]
    data6 = data[5]

    img1 = cv2.imread(input1_path)
    img1, with_defect = img_vis(img1, data1)
    if with_defect:
        img1_vis_path = os.path.join(os.path.dirname(input1_path)+'_vis', os.path.basename(input1_path))
        os.makedirs(os.path.dirname(img1_vis_path), exist_ok=True)
        cv2.imwrite(img1_vis_path, img1)

    img2 = cv2.imread(input2_path)
    img2, with_defect = img_vis(img2, data2)
    if with_defect:
        img2_vis_path = os.path.join(os.path.dirname(input2_path)+'_vis', os.path.basename(input2_path))
        os.makedirs(os.path.dirname(img2_vis_path), exist_ok=True)
        cv2.imwrite(img2_vis_path, img2)

    img3 = cv2.imread(input3_path)
    img3, with_defect = img_vis(img3, data3)
    if with_defect:
        img3_vis_path = os.path.join(os.path.dirname(input3_path)+'_vis', os.path.basename(input3_path))
        os.makedirs(os.path.dirname(img3_vis_path), exist_ok=True)
        cv2.imwrite(img3_vis_path, img3)

    img4 = cv2.imread(input4_path)
    img4, with_defect = img_vis(img4, data4)
    if with_defect:
        img4_vis_path = os.path.join(os.path.dirname(input4_path)+'_vis', os.path.basename(input4_path))
        os.makedirs(os.path.dirname(img4_vis_path), exist_ok=True)
        cv2.imwrite(img4_vis_path, img4)

    img5 = cv2.imread(input5_path)
    img5, with_defect = img_vis(img5, data5)
    if with_defect:
        img5_vis_path = os.path.join(os.path.dirname(input5_path)+'_vis', os.path.basename(input5_path))
        os.makedirs(os.path.dirname(img5_vis_path), exist_ok=True)
        cv2.imwrite(img5_vis_path, img5)

    img6 = cv2.imread(input6_path)
    img6, with_defect = img_vis(img6, data6)
    if with_defect:
        img6_vis_path = os.path.join(os.path.dirname(input6_path)+'_vis', os.path.basename(input6_path))
        os.makedirs(os.path.dirname(img6_vis_path), exist_ok=True)
        cv2.imwrite(img6_vis_path, img6)


def img_vis(img, results):
    height, width = img.shape[:2]
    risk_sum = 0
    for result in results:
        cat = result['category']
        risk_a = result['risk_a']
        risk_b = result['risk_b']
        risk_c = result['risk_c']
        risk_d = result['risk_d']
        risk_sum += risk_a + risk_b + risk_c + risk_d
        defect_str = ''
        if risk_a>0:
            defect_str += 'riskA;'
        if risk_b>0:
            defect_str += 'riskB;'
        if risk_c>0:
            defect_str += 'riskC;'
        if risk_d>0:
            defect_str += 'riskD;'
        uvs = result['uvs']
        color = COLOR_PALETTE[cat]
        uvs = np.array(uvs, dtype=np.float32)
        pixel_coords = (uvs * [width, height]).astype(np.int32)
        cv2.polylines(
            img,
            [pixel_coords],
            isClosed=True,
            color=(color[0], color[1], color[2], 255),  # BGR+Alpha（红色实线）
            thickness=2,
            lineType=cv2.LINE_AA
        )
        u_values = [uv[0] for uv in uvs]
        v_values = [uv[1] for uv in uvs]
        min_u = min(u_values)
        min_v = min(v_values)
        top_left_x = min_u*width
        top_left_y = min_v*height
        cv2.putText(img, defect_str, (int(top_left_x), int(top_left_y) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if risk_sum == 0:
        with_defect = False
    else:
        with_defect = True
    return img, with_defect


if __name__ == '__main__':
    pass
    # DA5324655_20250709161321600
    # DA5324655_20250709161321800
    dir_vis(
        json_dir=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_infer\seg',
        input_dir1=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_1',
        input_dir2=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_2',
        input_dir3=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_3',
        input_dir4=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_4',
        input_dir5=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_5',
        input_dir6=r'E:\cp_dir\0289d6677a5546b5aa6f256154c6cd23\input_6'
    )