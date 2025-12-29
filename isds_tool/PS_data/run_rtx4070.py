from typing import Any, NoReturn


import os
import shutil
from tqdm import tqdm
from yolo_mask_crop import *
from yolo2xanylabeling import yolo_to_xanylabeling_dir
from zip_tools import zip_folder_to_path
from data_vis.yolo_vis import yolo_mdet_vis

def att_check(input_dir, output_dir, reorder=False, rm_id=True):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    label_list = os.listdir(input_dir)
    track_list = []
    rm_id_list = []
    for label_name in tqdm(label_list):
        rm_id_flag=False
        input_label_path = os.path.join(input_dir, label_name)
        output_label_path = os.path.join(output_dir, label_name)
        with open(input_label_path, 'r') as f1, open(output_label_path, 'w') as f2:
            lines = f1.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                parts = line.strip().split(' ')
                risk_list = parts[2:6]
                if reorder:
                    new_risk_list = [risk_list[3], risk_list[1], risk_list[0], risk_list[2]]
                    parts[2:6] = new_risk_list
                if rm_id:
                    if len(parts) % 2 == 1:
                        parts = parts[:-1]
                        track_list.append(label_name)
                        rm_id_flag = True
                new_line = ' '.join(parts) + '\n'
                if line[2] == '0':
                    lines[idx] = lines[idx][0:2] + '4' + lines[idx][3:]
                    count += 1
                new_lines.append(new_line)
                if rm_id_flag:
                    rm_id_list.append(idx)
            f2.writelines(new_lines)
        with open(input_label_path, 'w') as f:
            f.writelines(lines)
    print(f'change {count} lines')
    track_list = list(set(track_list))
    print(len(track_list), track_list)
    print(f'rm id ({len(rm_id_list)}): {rm_id_list}')


def find_repeated_file(input_dir1, input_dir2):
    input_dir1_list = os.listdir(input_dir1)
    input_dir2_list = os.listdir(input_dir2)
    repeated_file_list = []
    for input_path1 in tqdm(input_dir1_list):
        if input_path1 in input_dir2_list:
            repeated_file_list.append(input_path1)
    print(f'repeated file list: {repeated_file_list}')


def remove_repeat(input_dir, ref_dir_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_file_list = os.listdir(input_dir)
    ref_file_list = []
    for ref_dir in ref_dir_list:
        ref_file_list += os.listdir(ref_dir)
    for input_name in tqdm(input_file_list):
        if input_name not in ref_file_list:
            continue
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, input_name)
        shutil.move(input_path, output_path)

def get_img_stem_2_obj_name(input_dir):
    obj_name_list = os.listdir(input_dir)
    img_stem_2_obj_name = {}
    for obj_name in tqdm(obj_name_list):
        if obj_name.endswith('.db'):
            continue
        img_stem,_ = obj_name.rsplit('_', 1)
        if img_stem in img_stem_2_obj_name:
            img_stem_2_obj_name[img_stem].append(obj_name)
        else:
            img_stem_2_obj_name[img_stem] = [obj_name]
    return img_stem_2_obj_name

def rm_obj_by_ref(input_dir, input_path):
    df = pd.read_csv(input_path, index_col=None, names=['file_path'])
    file_list = df['file_path'].tolist()
    file_stem_list = [Path(file_path).stem for file_path in file_list]
    risk = ['abandonment', 'broken', 'corrosion', 'deformation']
    levels = ['no', 'medium', 'high']
    all_list = []
    for risk in risk:
        for level in levels:
            risk_level_list = []
            risk_level_dir = os.path.join(input_dir, risk, level)
            img_stem_2_obj_name = get_img_stem_2_obj_name(risk_level_dir)
            for file_stem in file_stem_list:
                if file_stem in img_stem_2_obj_name:
                    for obj_name in img_stem_2_obj_name[file_stem]:
                        obj_path = os.path.join(risk_level_dir, obj_name)
                        risk_level_list.append(obj_path)
            all_list += risk_level_list
            print(f'find {len(risk_level_list)} in {risk} {level}')
    print(f'find {len(all_list)} total, unique {len(set(all_list))}')
    for file_path in tqdm(all_list):
        os.remove(file_path)

if __name__ == '__main__':
    pass

    yolo_to_xanylabeling_dir(
        yolo_label_dir=r'\\158.132.186.40\isds\huilin\isds\other_data\1118_copy\synthetic_data_add6_v1_infer_b\labels',
        images_dir=r'\\158.132.186.40\isds\huilin\isds\other_data\1118_copy\synthetic_data_add6_v1',
        xanylabeling_label_dir=r'\\158.132.186.40\isds\huilin\isds\other_data\1118_copy\synthetic_data_add6_v1_json',
        class_file=r'\\158.132.186.40\isds\huilin\isds\check_data\class.txt',
        attribute_file=r'\\158.132.186.40\isds\huilin\isds\other_data\1118_copy\attribute_b.yaml',
    )

    # input_dir = r'\\158.132.186.40\isds\huilin\isds\other_data\check1113\data7961'
    # input_path = r'E:\cp_dir\val_test.txt'
    # rm_obj_by_ref(input_dir, input_path)
    # input_dir = r'E:\cp_dir\result_analysis\select_labels_0919\select_labels_0919'
    # output_dir1 = r'E:\cp_dir\result_analysis\select_labels_0919\labels'
    # # att_check(input_dir, output_dir)
    # input_dir = r'E:\cp_dir\result_analysis\selset_pre_labels_0921\selset_pre_labels_0921'
    # output_dir2 = r'E:\cp_dir\result_analysis\selset_pre_labels_0921\labels'
    # # att_check(input_dir, output_dir2)
    # find_repeated_file(output_dir1, output_dir2)

    # root_dir = r'E:\cp_dir\result_analysis\selset_pre_labels_0921'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'labels')
    # image_crop_dir = os.path.join(dataset_dir, 'images_crop')
    # class_file = os.path.join(dataset_dir, 'class_c6.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')


    # root_dir = r'\\158.132.186.40\isds\huilin\isds\other_data\task1008'
    # image_dir = os.path.join(root_dir, 'merge_dir')
    # label_dir = os.path.join(root_dir, 'merge_dir_infer', 'labels')
    # json_dir = os.path.join(root_dir, 'json')
    # class_file = os.path.join(root_dir, 'class_c6.txt')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # # yolo_to_xanylabeling_dir(label_dir, image_dir, json_dir, class_file, attribute_file)
    #
    # zip_folder_to_path(
    #     source_folder=image_dir,
    #     destination_zip=os.path.join(root_dir, os.path.basename(root_dir)+'.zip')
    # )
    # zip_folder_to_path(
    #     source_folder=label_dir,
    #     destination_zip=os.path.join(root_dir, os.path.basename(root_dir)+'_labels.zip')
    # )
    # zip_folder_to_path(
    #     source_folder=json_dir,
    #     destination_zip=os.path.join(root_dir, os.path.basename(root_dir)+'_jsons.zip')
    # )

    # root_dir = r'\\158.132.186.40\isds\huilin\isds\other_data\check1021'
    # images_dir = os.path.join(root_dir, 'images')
    # labels_dir1 = os.path.join(root_dir, 'check1021_labels_1022')
    # labels_dir2 = os.path.join(root_dir, 'check1021_labels_1022_re')
    # images_vis_dir = os.path.join(root_dir, 'images_vis')
    # image_crop_dir = os.path.join(root_dir, 'images_crop')
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class.txt')
    # att_check(labels_dir1, labels_dir2, reorder=False, rm_id=True)
    # myolo_crop(images_dir, labels_dir2, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')
    # yolo_mdet_vis(images_dir, labels_dir2, images_vis_dir, class_file, crop_dir=None, seg=True,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=False, seg_crop=False)

    # remove_repeat(
    #     input_dir=r'E:\tp\broken_medium_1020',
    #     ref_dir_list=[
    #         r'E:\tp\broken_high_1020',
    #         r'E:\tp\crack_high_1020',
    #         r'E:\tp\crack_medium_1020',
    #         r'E:\tp\hole_high_1020',
    #         r'E:\tp\hole_medium1020',
    #         r'E:\tp\no_1020',
    #     ],
    #     output_dir=r'E:\tp\repeat'
    # )