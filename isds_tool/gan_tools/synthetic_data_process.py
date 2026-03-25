import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(r'E:\repository\dataset_tools')
from isds_tool.PS_data.yolo_tools import get_yolo_label_df, poly2xyxy
import onnxruntime as ort


class FastReID_ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, img):
        if isinstance(img, str):
            img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]
        return img

    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def extract(self, patch):
        input_tensor = self.preprocess(patch)
        feat = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0]
        feat = self.normalize(feat, axis=1)
        return feat

def file_stem2name(input_dir):
    name_list = os.listdir(input_dir)
    stem_list = [Path(name).stem for name in name_list]
    stem2name_dict = {stem: name for stem, name in zip(stem_list, name_list)}
    return stem2name_dict


def merge_object2image(image_path, object_path_list, result_path, boxes):
    image = cv2.imread(image_path)
    for object_path, box in zip(object_path_list, boxes):
        obj = cv2.imread(object_path)
        h, w, _ = image.shape
        box = [int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)]
        obj_rs = cv2.resize(obj, (box[2]-box[0], box[3]-box[1]))
        image[box[1]:box[3], box[0]:box[2]] = obj_rs
    cv2.imwrite(result_path, image)

def generate_object2image(image_path, object_path_list, result_path, boxes):
    image = cv2.imread(image_path)
    image_empty = np.zeros_like(image)
    for object_path, box in zip(object_path_list, boxes):
        obj = cv2.imread(object_path)
        h, w, _ = image.shape
        box = [int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)]
        obj_rs = cv2.resize(obj, (box[2]-box[0], box[3]-box[1]))
        image_empty[box[1]:box[3], box[0]:box[2]] = obj_rs
    cv2.imwrite(result_path, image_empty)

def xywh2xyxy(box):
    x1, y1, w, h = box
    x2, y2 = x1+w, y1+h
    return [x1, y1, x2, y2]

def label_update(input_path, output_path, update_idxes):
    update_idxes = [int(update_idx) for update_idx in update_idxes]
    # c 4 d b a c
    boxes = []
    with open(input_path, 'r') as fr:
        lines = fr.readlines()
        new_lines = []
        for line_id, line in enumerate(lines):
            if line_id in update_idxes:
                parts = line.strip().split(' ')
                att_len = int(parts[1])
                assert att_len == 4, f'Error, att_len should be 4, but got {att_len}'

                parts[3] = '1'
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

                polygons = list(map(float, parts[2 + att_len:]))
                box = poly2xyxy(polygons)
                boxes.append(box)
            else:
                new_lines.append(line)
        with open(output_path, 'w') as fw:
            fw.writelines(new_lines)
    return boxes

def label_extract(input_path, output_path, update_idxes):
    update_idxes = [int(update_idx) for update_idx in update_idxes]
    # c 4 d b a c
    boxes = []
    with open(input_path, 'r') as fr:
        lines = fr.readlines()
        new_lines = []
        for line_id, line in enumerate(lines):
            if line_id in update_idxes:
                parts = line.strip().split(' ')
                att_len = int(parts[1])
                assert att_len == 4, f'Error, att_len should be 4, but got {att_len}'

                parts[3] = '1'
                new_line = ' '.join(parts) + '\n'
                new_lines.append(new_line)

                polygons = list(map(float, parts[2 + att_len:]))
                box = poly2xyxy(polygons)
                boxes.append(box)
        with open(output_path, 'w') as fw:
            fw.writelines(new_lines)
    return boxes

def result_merge(result_dir, input_dir, output_dir):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels_v18')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    input_image_stem2name_dict = file_stem2name(input_image_dir)

    update_img_stem = {}
    result_file_list = os.listdir(result_dir)
    for result_file_name in tqdm(result_file_list, desc='get obj result'):
        file_stem, obj_id = Path(result_file_name).stem.rsplit('_', 1)
        input_image_name = input_image_stem2name_dict[file_stem]
        if file_stem not in update_img_stem:
            update_img_stem[file_stem] = [result_file_name]
        else:
            update_img_stem[file_stem].append(result_file_name)

    for idx, (file_stem, result_file_name_list) in enumerate(update_img_stem.items()):
        print(f'process {idx}/{len(update_img_stem)}: {file_stem}')
        input_image_name = input_image_stem2name_dict[file_stem]
        file_output_stem = file_stem+'_synthetic'
        output_image_name = file_output_stem+'.jpg'
        output_label_name = file_output_stem+'.txt'
        input_image_path = os.path.join(input_image_dir, input_image_name)
        input_label_path = os.path.join(input_label_dir, f'{file_stem}.txt')
        output_image_path = os.path.join(output_image_dir, output_image_name)
        output_label_path = os.path.join(output_label_dir, output_label_name)

        obj_ids = []
        result_file_path_list = []
        for result_file_name in result_file_name_list:
            file_stem, obj_id = Path(result_file_name).stem.rsplit('_', 1)
            obj_ids.append(obj_id)
            result_file_path = os.path.join(result_dir, result_file_name)
            result_file_path_list.append(result_file_path)

        boxes = label_update(input_label_path, output_label_path, obj_ids)
        merge_object2image(input_image_path, result_file_path_list, output_image_path, boxes)


def _cosine_sim(a, b):
    return np.dot(a, b.T)

def result_match(check_dir_src, check_dir_result, check_dir_dst, reid_path):
    os.makedirs(check_dir_dst, exist_ok=True)
    reid_session = FastReID_ONNX(reid_path)
    src_obj_list = os.listdir(check_dir_src)
    result_obj_list = os.listdir(check_dir_result)
    src_obj_path_list = [os.path.join(check_dir_src, src_obj_name) for src_obj_name in src_obj_list]
    result_obj_path_list = [os.path.join(check_dir_result, result_obj_name) for result_obj_name in result_obj_list]

    src_embedding_list = [reid_session.extract(src_obj_path) for src_obj_path in tqdm(src_obj_path_list, desc='extract src')]
    result_embedding_list = [reid_session.extract(result_obj_path) for result_obj_path in tqdm(result_obj_path_list, desc='extract result')]
    src_embedding_list = np.concatenate(src_embedding_list, axis=0)
    result_embedding_list = np.concatenate(result_embedding_list, axis=0)
    sim_matrix = _cosine_sim(src_embedding_list, result_embedding_list)


    max_sim_index = np.argmax(sim_matrix, axis=1)
    max_sim_value = np.max(sim_matrix, axis=1)

    match_dict = {}
    for i, (max_index, max_value) in enumerate(zip(max_sim_index, max_sim_value)):
        if max_value > 0.5:
            match_dict[src_obj_list[i]] = result_obj_list[max_index]
            src_obj_path = src_obj_path_list[i]
            result_obj_path = result_obj_path_list[max_index]
            dst_obj_path = os.path.join(check_dir_dst, src_obj_list[i])
            shutil.copy(result_obj_path, dst_obj_path)
        else:
            match_dict[src_obj_list[i]] = None
    
    print(match_dict)

def result_generate(result_dir, input_dir, output_dir):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels_v18')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    input_image_stem2name_dict = file_stem2name(input_image_dir)

    update_img_stem = {}
    result_file_list = os.listdir(result_dir)
    for result_file_name in tqdm(result_file_list, desc='get obj result'):
        file_stem, obj_id = Path(result_file_name).stem.rsplit('_', 1)
        input_image_name = input_image_stem2name_dict[file_stem]
        if file_stem not in update_img_stem:
            update_img_stem[file_stem] = [result_file_name]
        else:
            update_img_stem[file_stem].append(result_file_name)

    for idx, (file_stem, result_file_name_list) in enumerate(update_img_stem.items()):
        print(f'process {idx}/{len(update_img_stem)}: {file_stem}')
        input_image_name = input_image_stem2name_dict[file_stem]
        file_output_stem = file_stem+'_synthetic_empty'
        output_image_name = file_output_stem+'.jpg'
        output_label_name = file_output_stem+'.txt'
        input_image_path = os.path.join(input_image_dir, input_image_name)
        input_label_path = os.path.join(input_label_dir, f'{file_stem}.txt')
        output_image_path = os.path.join(output_image_dir, output_image_name)
        output_label_path = os.path.join(output_label_dir, output_label_name)

        obj_ids = []
        result_file_path_list = []
        for result_file_name in result_file_name_list:
            file_stem, obj_id = Path(result_file_name).stem.rsplit('_', 1)
            obj_ids.append(obj_id)
            result_file_path = os.path.join(result_dir, result_file_name)
            result_file_path_list.append(result_file_path)

        boxes = label_extract(input_label_path, output_label_path, obj_ids)
        generate_object2image(input_image_path, result_file_path_list, output_image_path, boxes)

if __name__ == '__main__':
    input_data_dir = r'\\158.132.186.40\isds\huilin\isds\copy\data7961_mseg_c6_1030'
    # output_data_dir = r'\\158.132.186.40\isds\huilin\isds\check_data\synthetic_data_add1_v1'
    # src_obj_dir = r'E:\cp_dir\synthetic_data_add1_v1'
    # dst_obj_dir0 = r'E:\cp_dir\synthetic_data_add1_v1_result0'
    # dst_obj_dir1 = r'E:\cp_dir\synthetic_data_add1_v1_result1'

    # data_name = 'synthetic_data_add2_v1'
    # output_data_dir = os.path.join(r'\\158.132.186.40\isds\huilin\isds\check_data', data_name)
    # src_obj_dir = os.path.join(r'E:\cp_dir', data_name)
    # dst_obj_dir0 = os.path.join(r'E:\cp_dir', data_name+'_result0')
    # dst_obj_dir1 = os.path.join(r'E:\cp_dir', data_name+'_result1')
    # reid_path = r'E:\repository\dataset_tools\isds_tool\gan_tools\reid_model.onnx'
    # result_match(src_obj_dir, dst_obj_dir0, dst_obj_dir1, reid_path)
    # result_merge(dst_obj_dir1, input_data_dir, output_data_dir)
    # result_generate(dst_obj_dir1, input_data_dir, output_data_dir)

    data_name = 'synthetic_data_add4_v1'
    output_data_dir = os.path.join(r'\\158.132.186.40\isds\huilin\isds\check_data', data_name)
    src_obj_dir = os.path.join(r'E:\cp_dir', data_name)
    reid_path = r'E:\repository\dataset_tools\isds_tool\gan_tools\reid_model.onnx'
    result_merge(src_obj_dir, input_data_dir, output_data_dir)
    result_generate(src_obj_dir, input_data_dir, output_data_dir)