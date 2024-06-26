'''
Created on Aug 18, 2021

@author: xiaosonh
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
import pandas as pd

from sklearn.model_selection import train_test_split
from labelme import utils


class Labelme2YOLO(object):

    def __init__(self, json_dir, save_path, to_seg=False, multi_task=False):
        self._json_dir = json_dir

        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg
        self._multi_task = multi_task

        i = 'YOLODataset'
        i += '_seg/' if to_seg else '/'
        self._save_path = save_path

    def _make_train_val_dir(self, tv=True):
        self._label_dir_path = os.path.join(self._save_path, 'labels/')
        self._image_dir_path = os.path.join(self._save_path, 'images/')
        if not os.path.exists(self._label_dir_path):
            os.makedirs(self._label_dir_path)
        if not os.path.exists(self._image_dir_path):
            os.makedirs(self._image_dir_path)
        if tv:
            for yolo_path in (os.path.join(self._label_dir_path + 'train/'),
                              os.path.join(self._label_dir_path + 'val/'),
                              os.path.join(self._image_dir_path + 'train/'),
                              os.path.join(self._image_dir_path + 'val/')):
                if os.path.exists(yolo_path):
                    shutil.rmtree(yolo_path)

                os.makedirs(yolo_path)

    def _get_label_id_map(self, json_dir):
        label_set = set()

        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])

        return OrderedDict([(label, label_id) \
                            for label_id, label in enumerate(label_set)])

    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]

            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]

            return train_json_names, val_json_names
        if val_size>0:
            train_idxs, val_idxs = train_test_split(range(len(json_names)),
                                                    test_size=val_size)
            train_json_names = [json_names[train_idx] for train_idx in train_idxs]
            val_json_names = [json_names[val_idx] for val_idx in val_idxs]

            return train_json_names, val_json_names
        else:
            return json_names, []

    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders = [file_name for file_name in os.listdir(self._json_dir) \
                   if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)

        self._make_train_val_dir()

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train/', 'val/'),
                                          (train_json_names, val_json_names)):
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))

                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))

                img_path = self._save_yolo_image(json_data,
                                                 json_name,
                                                 self._image_dir_path,
                                                 target_dir)

                yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
                self._save_yolo_label(json_name,
                                      self._label_dir_path,
                                      target_dir,
                                      yolo_obj_list)

        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
    def convert_all(self):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders = [file_name for file_name in os.listdir(self._json_dir) \
                   if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, _ = self._train_test_split(folders, json_names, 0)

        self._make_train_val_dir(tv=False)

        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        # for target_dir, json_names in zip(('train/', 'val/'),
        #                                   (train_json_names, val_json_names)):

        for json_name in train_json_names:
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))

            print('Converting %s ...' % (json_name))

            img_path = self._save_yolo_image(json_data,
                                             json_name,
                                             self._image_dir_path,
                                             '')

            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_name,
                                  self._label_dir_path,
                                  '',
                                  yolo_obj_list)

        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()

    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))

        print('Converting %s ...' % json_name)

        img_path = self._save_yolo_image(json_data, json_name,
                                         self._json_dir, '')

        yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
        self._save_yolo_label(json_name, self._json_dir,
                              '', yolo_obj_list)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []

        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            # labelme circle shape is different from others
            # it only has 2 points, 1st is circle center, 2nd is drag end point
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)

            yolo_obj_list.append(yolo_obj)

        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]

        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]

            n_part = radius / 10
            n_part = int(n_part) if n_part > 4 else 4
            n_part2 = n_part << 1

            pt_quad = [None for i in range(0, 4)]
            pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) * radius,
                           obj_center_y - math.sin(i * math.pi / n_part2) * radius]
                          for i in range(1, n_part)]
            pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
            pt_quad[1].reverse()
            pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
            pt_quad[3].reverse()
            pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
            pt_quad[2].reverse()

            pt_quad[0].append([obj_center_x, obj_center_y - radius])
            pt_quad[1].append([obj_center_x - radius, obj_center_y])
            pt_quad[2].append([obj_center_x, obj_center_y + radius])
            pt_quad[3].append([obj_center_x + radius, obj_center_y])

            for i in pt_quad:
                for j in i:
                    j[0] = round(float(j[0]) / img_w, 6)
                    j[1] = round(float(j[1]) / img_h, 6)
                    retval.extend(j)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius

        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]

        if self._to_seg:
            retval = [label_id]
            if shape['shape_type'] == 'rectangle' and not self._multi_task:
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                new_points = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                for i in new_points:
                    i[0] = round(float(i[0]) / img_w, 6)
                    i[1] = round(float(i[1]) / img_h, 6)
                    retval.extend(i)
            if shape['shape_type'] == 'rectangle' and self._multi_task:
                for i in shape['points']:
                    i[0] = round(float(i[0]) / img_w, 6)
                    i[1] = round(float(i[1]) / img_h, 6)
                    retval.extend(i)
            else:
                for i in shape['points']:
                    i[0] = round(float(i[0]) / img_w, 6)
                    i[1] = round(float(i[1]) / img_h, 6)
                    retval.extend(i)
            return retval

        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)

            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]

            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)

        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path,
                                target_dir,
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = ""
                for i in yolo_obj:
                    yolo_obj_line += f'{i} '
                yolo_obj_line = yolo_obj_line[:-1]
                if yolo_obj_idx != len(yolo_obj_list) - 1:
                    yolo_obj_line += '\n'
                f.write(yolo_obj_line)

    def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(image_dir_path, target_dir, img_name)

        if not os.path.exists(img_path):
            img = utils.img_b64_to_arr(json_data['imageData'])
            PIL.Image.fromarray(img).save(img_path)

        return img_path

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._save_path, 'dataset.yaml')

        with open(yaml_path, 'w+') as yaml_file:
            yaml_file.write('train: %s\n' % \
                            os.path.join(self._image_dir_path, 'train/'))
            yaml_file.write('val: %s\n\n' % \
                            os.path.join(self._image_dir_path, 'val/'))
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))

            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)

    def save_class(self, save_path):
        class_list = list(self._label_id_map.keys())
        df = pd.DataFrame(class_list, columns=['category'])
        df.to_csv(save_path, index=False, header=False)

if __name__ == '__main__':
    pass

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--json_dir', type=str,
    #                     help='Please input the path of the labelme json files.')
    # parser.add_argument('--val_size', type=float, nargs='?', default=0.1,
    #                     help='Please input the validation dataset size, for example 0.1 ')
    # parser.add_argument('--json_name', type=str, nargs='?', default=None,
    #                     help='If you put json name, it would convert only one json file to YOLO.')
    # parser.add_argument('--seg', action='store_true',
    #                     help='Convert to YOLOv5 v7.0 segmentation dataset')
    # args = parser.parse_args(sys.argv[1:])
    #
    # convertor = Labelme2YOLO(args.json_dir, to_seg=args.seg)
    # if args.json_name is None:
    #     convertor.convert(val_size=args.val_size)
    # else:
    #     convertor.convert_one(args.json_name)

    # json_dir = r'E:\data\0417_signboard\data0417\labelme\labels'
    # save_dir = r'E:\data\0417_signboard\data0417\yolo'
    #
    # convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    # convertor.convert(val_size=0.0)


    # json_dir = r'E:\data\0417_signboard\data0420\labelme'
    # save_dir = r'E:\data\0417_signboard\data0420\yolo'
    # convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    # convertor.convert_all()

    # json_dir = r'E:\data\0417_signboard\data0507\norm\rgb'
    # save_dir = r'E:\data\0417_signboard\data0507\yolo'
    # convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    # convertor.convert_all()
    # convertor.save_class(r'E:\data\0417_signboard\data0507\yolo\class.txt')

    # json_dir = r'E:\data\0417_signboard\data0514\norm\rgb'
    # save_dir = r'E:\data\0417_signboard\data0514\yolo'
    # convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    # convertor._label_id_map = OrderedDict({'surface':0, 'frame':1})
    # print(convertor._label_id_map)
    # convertor.convert_all()
    # convertor.save_class(r'E:\data\0417_signboard\data0514\yolo\class.txt')

    # json_dir = r'E:\data\0417_signboard\data0521_m\norm\rgb_detection1'
    # save_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection1'
    # convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    # convertor._label_id_map = OrderedDict({'surface':0, 'frame':1, 'normal':2, 'incomplete':3})
    # convertor.convert_all()
    # convertor.save_class(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection1\class.txt')

    json_dir = r'E:\data\0417_signboard\data0521_m\norm\rgb_detection2'
    save_dir = r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection2'
    convertor = Labelme2YOLO(json_dir, save_dir, to_seg=True)
    convertor._label_id_map = OrderedDict({'surface':0,
                                           'frame':1,
                                           'normal':2,
                                           'obstructed':3,
                                           'missing': 4,
                                           'incomplete2': 5,
                                           'peeling2': 6,
                                           'fade': 7,
                                           'deformed': 8,
                                           'corroded2': 9,
                                           })
    convertor.convert_all()
    convertor.save_class(r'E:\data\0417_signboard\data0521_m\yolo_rgb_detection2\class.txt')