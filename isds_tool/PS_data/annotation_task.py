import os
import shutil
from pathlib import Path
from tqdm import tqdm

def get_task_batch(input_img_dir, input_pred_dir, input_json_dir, output_dir, class_path, k=2):
    pass
    input_pred_txt_dir = input_pred_dir
    output_dir_list = [f'{output_dir}_{i+1}' for i in range(k)]
    output_image_dir_list = [os.path.join(output_dir, 'image') for output_dir in output_dir_list]
    output_label_dir_list = [os.path.join(output_dir, 'label') for output_dir in output_dir_list]
    output_json_dir_list = [os.path.join(output_dir, 'json') for output_dir in output_dir_list]
    print(output_label_dir_list)
    print(output_json_dir_list)


    for output_image_dir in output_image_dir_list:
        os.makedirs(output_image_dir, exist_ok=True)
    for output_label_dir in output_label_dir_list:
        os.makedirs(output_label_dir, exist_ok=True)
        shutil.copy(class_path, os.path.join(os.path.dirname(output_label_dir), 'class.txt'))
    for output_json_dir in output_json_dir_list:
        os.makedirs(output_json_dir, exist_ok=True)
        print(output_json_dir)
    file_list = os.listdir(input_img_dir)
    for i in tqdm(range(0, len(file_list), k)):
        for j in range(k):
            idx = i + j
            if idx < len(file_list):
                image_name = file_list[idx]
                label_name = Path(file_list[idx]).stem + '.txt'
                json_name = Path(file_list[idx]).stem + '.json'
                input_img_path = os.path.join(input_img_dir, image_name)
                input_pred_txt_path = os.path.join(input_pred_txt_dir, label_name)
                input_json_path = os.path.join(input_json_dir, json_name)
                output_img_path = os.path.join(output_image_dir_list[j], image_name)
                ouput_pred_txt_path = os.path.join(output_label_dir_list[j], label_name)
                output_json_path = os.path.join(output_json_dir_list[j], json_name)
                shutil.copyfile(input_img_path, output_img_path)
                if os.path.exists(input_pred_txt_path):
                    shutil.copyfile(input_pred_txt_path, ouput_pred_txt_path)
                if os.path.exists(input_json_path):
                    shutil.copyfile(input_json_path, output_json_path)


if __name__ == '__main__':
    pass
    CLASS_PATH = r'E:\data\202502_signboard\data_annotation\annotation guide 0510\class.txt'
    # input_img_dir=r'E:\data\202502_signboard\data_annotation\task\task0519\images_split\left'
    # input_pred_dir=r'E:\data\202502_signboard\data_annotation\task\task0519\images_split_pred\left'
    # output_dir = r'E:\data\202502_signboard\data_annotation\task\task0519\ps_task_left_batch'
    # get_task_batch(input_img_dir, input_pred_dir, output_dir, CLASS_PATH)
    # input_img_dir=r'E:\data\202502_signboard\data_annotation\task\task0519\images_split\right'
    # input_pred_dir=r'E:\data\202502_signboard\data_annotation\task\task0519\images_split_pred\right'
    # output_dir = r'E:\data\202502_signboard\data_annotation\task\task0519\ps_task_right_batch'
    # get_task_batch(input_img_dir, input_pred_dir, output_dir, CLASS_PATH)

    input_img_dir=r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\images_updated'
    input_pred_dir=r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\labels_updated'
    output_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\pseudo_label_data\ps_task_batch'
    get_task_batch(input_img_dir, input_pred_dir, output_dir, CLASS_PATH)