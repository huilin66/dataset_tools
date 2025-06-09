import os
import shutil
from tqdm import tqdm
from pathlib import Path

def copy_imgs(dst_dir, ref_dir):
    dst_images_dir = os.path.join(dst_dir, 'images')
    ref_images_dir = os.path.join(ref_dir, 'images')
    if not os.path.exists(dst_images_dir):
        print(f'copying images from {ref_images_dir} to {dst_images_dir}')
        shutil.copytree(ref_images_dir, dst_images_dir)


def files_check():
    pass

def anno2data(annotations_dir, dataset_dir, class_path, attribute_path):
    anno_list = os.listdir(annotations_dir)
    anno_list.sort()
    data_image_dir = os.path.join(dataset_dir, 'images')
    data_label_dir = os.path.join(dataset_dir, 'labels')
    os.makedirs(data_image_dir, exist_ok=True)
    os.makedirs(data_label_dir, exist_ok=True)
    for idx, anno_name in enumerate(anno_list):
        anno_dir = os.path.join(annotations_dir, anno_name)
        ref_dir = os.path.join(os.path.dirname(annotations_dir)[:-5], anno_name[:-5])
        copy_imgs(anno_dir, ref_dir)
        images_dir = os.path.join(anno_dir, 'images')
        labels_dir = os.path.join(anno_dir, 'labels')
        images_list = os.listdir(images_dir)
        for image_name in tqdm(images_list, desc=f'merge {anno_name}; {idx}/{len(anno_list)}'):
            if 'left' in anno_name:
                dst_image_name = Path(image_name).stem + '_left'+Path(image_name).suffix
            elif 'right' in anno_name:
                dst_image_name = Path(image_name).stem + '_right'+Path(image_name).suffix
            else:
                dst_image_name = image_name
            label_name = Path(image_name).stem + '.txt'
            dst_label_name = Path(dst_image_name).stem + '.txt'
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                continue
            dst_image_path = os.path.join(data_image_dir, dst_image_name)
            dst_label_path = os.path.join(data_label_dir, dst_label_name)
            shutil.copy(image_path, dst_image_path)
            shutil.copy(label_path, dst_label_path)
    shutil.copy(class_path, os.path.join(dataset_dir, 'class.txt'))
    shutil.copy(attribute_path, os.path.join(dataset_dir, 'attribute.yaml'))

if __name__ == '__main__':
    pass
    annotations_dir = r'E:\data\202502_signboard\data_annotation\task\task0528_anno\annos'
    dataset_dir = r'E:\data\202502_signboard\data_annotation\task\task0528_anno\yolo_dataset'
    class_path = r'E:\data\202502_signboard\data_annotation\annotation guide 0510\class.txt'
    attribute_path = r'E:\data\202502_signboard\data_annotation\annotation guide 0510\attribute_v5.yaml'
    anno2data(annotations_dir, dataset_dir, class_path, attribute_path)
