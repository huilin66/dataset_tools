import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from isds_tool.PS_data.yolo_mask_crop import get_cats,get_atts


def get_obj_info_json(csv_path, json_dir):
    pass
    os.makedirs(json_dir, exist_ok=True)
    df = pd.read_csv(csv_path, header=0, index_col=0)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        save_name = Path(data['image_name_object']).stem + '.json'
        save_path = os.path.join(json_dir, save_name)
        with open(save_path, 'w') as f:
            f.write(json.dumps(data))

def get_caption(csv_path, caption_dir, class_file, attribute_file):
    pass
    cats = get_cats(class_file)
    cats = dict(zip(list(range(len(cats))), cats))
    atts = get_atts(attribute_file)
    levels = {0:'no', 1:'medium', 2:'high'}
    os.makedirs(caption_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['with_defect'] = (df['deformation'] > 0) | (df['broken'] > 0) | (df['abandonment'] > 0) | (df['corrosion'] > 0)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        data = row.to_dict()
        new_data = {'file_name': data['image_name_object']}
        if data['with_defect']:
            caption_str = f"This is a {cats[data['category']]} signboard with defect: abandonment is {levels[data['abandonment']]} risk, broken is {levels[data['broken']]} risk, corrosion is {levels[data['corrosion']]} risk, deformation is {levels[data['deformation']]} risk"
        else:
            caption_str = f"This is a {cats[data['category']]} signboard without defect."
        new_data['text'] = caption_str

        save_name = Path(data['image_name_object']).stem + '.json'
        save_path = os.path.join(caption_dir, save_name)
        with open(save_path, 'w') as f:
            f.write(json.dumps(new_data))

def json2jsonl(input_dir, output_path):
    file_list = os.listdir(input_dir)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for file_name in tqdm(file_list):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)  # 读取JSON数据

                # 3. 处理数据（可能是数组或单个对象）
                if isinstance(data, list):
                    # 如果是JSON数组，逐行写入
                    for item in data:
                        json.dump(item, outfile, ensure_ascii=False)
                        outfile.write("\n")  # 换行分隔
                else:
                    # 如果是单个JSON对象，直接写入
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")

    print(f"合并完成！输出文件：{output_path}")


# 'deformation''broken''abandonment''corrosion'
def seg_label_to_sentence(line, class_names, atts, levels):
    parts = line.strip().split()
    cls_id = int(parts[0])
    risk_d, risk_b, risk_a, risk_c = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    with_risk = (risk_a + risk_b + risk_c + risk_d) > 0
    coords = list(map(float, parts[6:]))

    xs = coords[0::2]
    ys = coords[1::2]

    # 计算外接框中心
    x_c = (min(xs) + max(xs)) / 2
    y_c = (min(ys) + max(ys)) / 2

    # 横向位置
    if x_c < 1/3:
        col = "left"
    elif x_c < 2/3:
        col = "center"
    else:
        col = "right"

    # 纵向位置
    if y_c < 1/3:
        row = "top"
    elif y_c < 2/3:
        row = "middle"
    else:
        row = "bottom"

    cls_name = class_names[cls_id]
    if with_risk:
        caption = f"there is a {cls_name} signboard in {row}-{col} with defect: abandonment is {levels[risk_a]} risk, broken is {levels[risk_b]} risk, corrosion is {levels[risk_c]} risk, deformation is {levels[risk_d]} risk;"
    else:
        caption = f"there is a {cls_name} signboard in {row}-{col} without defect;"
    return caption

def get_caption_img(images_dir, label_dir, output_dir, class_path, attribute_path):
    cats = get_cats(class_path)
    cats = dict(zip(list(range(len(cats))), cats))
    atts = get_atts(attribute_path)
    levels = {0:'no', 1:'medium', 2:'high'}


    os.makedirs(output_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(images_dir)):
        label_name = Path(image_name).stem + '.txt'

        label_path = os.path.join(label_dir, label_name)
        output_path = os.path.join(output_dir, label_name.replace('.txt', '.json'))

        with open(label_path, "r") as f:
            lines = f.readlines()

        sentences = ''
        for line in lines:
            line = line.strip()
            sentence = seg_label_to_sentence(line, cats, atts, levels)
            sentences += sentence

        new_data = {'file_name': image_name, 'text': sentences}
        with open(output_path, "w") as f:
            f.write(json.dumps(new_data))

if __name__ == '__main__':
    pass
    # root_dir = r'/data/huilin/data/isds/fused_data/data3899_mseg_c6_0818/'
    # csv_path = os.path.join(root_dir, 'images_crop.csv')
    # json_dir = os.path.join(root_dir, 'images_crop_json_caption')
    # json_path = os.path.join(root_dir, 'images_crop_caption.jsonl')
    # class_path = os.path.join(root_dir, 'class.txt')
    # attribute_path = os.path.join(root_dir, 'attribute.yaml')
    # get_caption(csv_path, json_dir, class_path, attribute_path)
    # json2jsonl(json_dir, json_path)

    root_dir = r'/data/huilin/data/isds/fused_data/data3899_mseg_c6_0818/'
    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')
    image_caption_dir = os.path.join(root_dir, 'image_captions')
    image_caption_path = os.path.join(root_dir, 'image_captions.jsonl')
    class_path = os.path.join(root_dir, 'class.txt')
    attribute_path = os.path.join(root_dir, 'attribute.yaml')
    # get_caption_img(image_dir, label_dir, image_caption_dir, class_path, attribute_path)
    json2jsonl(image_caption_dir, image_caption_path)