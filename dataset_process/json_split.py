# -*- coding: utf-8 -*-
# @File             : json_split.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2021/8/1 10:25
# @Contributor      : zhaoHL
# @Time Modify Last : 2021/8/1 10:25
'''
@File Description:
# json数据集划分，可以通过val_split_rate、val_split_num控制划分比例或个数, keep_val_inTrain可以设定是否在train中保留val相关信息
'''
import shutil
import os
import json
import argparse
import pandas as pd

def get_annno(df_img_split, df_anno):
    df_merge = pd.merge(df_img_split, df_anno, on="image_id")
    df_anno_split = df_merge[df_anno.columns.to_list()]
    df_anno_split = df_anno_split.sort_values(by='id')
    return df_anno_split


def js_split(js_all_path, js_train_path, js_val_path, val_split_rate=0.1, val_split_num=None, keep_val_inTrain=False,
             img_split_path=None, img_src_path=None):
    print('Split'.center(100,'-'))
    print()

    print('json read...\n')

    with open(js_all_path, 'r') as load_f:
        data = json.load(load_f)
    df_anno = pd.DataFrame(data['annotations'])
    df_img = pd.DataFrame(data['images'])
    df_img = df_img.rename(columns={"id": "image_id"})
    df_img = df_img.sample(frac=1, random_state=0)

    if val_split_num is None:
        val_split_num = int(val_split_rate*len(df_img))

    if keep_val_inTrain:
        df_img_train = df_img
        df_img_val = df_img[: val_split_num]
        df_anno_train = df_anno
        df_anno_val = get_annno(df_img_val, df_anno)
    else:
        df_img_train = df_img[val_split_num:]
        df_img_val = df_img[: val_split_num]
        df_anno_train = get_annno(df_img_train, df_anno)
        df_anno_val = get_annno(df_img_val, df_anno)
    df_img_train = df_img_train.rename(columns={"image_id": "id"}).sort_values(by='id')
    df_img_val =df_img_val.rename(columns={"image_id": "id"}).sort_values(by='id')

    data['images'] = json.loads(df_img_train.to_json(orient='records'))
    data['annotations'] = json.loads(df_anno_train.to_json(orient='records'))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(js_train_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(str_json)

    data['images'] = json.loads(df_img_val.to_json(orient='records'))
    data['annotations'] = json.loads(df_anno_val.to_json(orient='records'))
    str_json = json.dumps(data, ensure_ascii=False)
    with open(js_val_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(str_json)

    if img_split_path is not None and img_src_path is not None:
        for idx, row in df_img_val.iterrows():
            da = row['file_name']
            input_path = os.path.join(img_src_path, row['file_name'])
            output_path = os.path.join(img_split_path, row['file_name'])
            if keep_val_inTrain:
                shutil.copy(input_path, output_path)
            else:
                shutil.move(input_path, output_path)

    print('image total %d, train %d, val %d'%(len(df_img), len(df_img_train), len(df_img_val)))
    print('anno total %d, train %d, val %d'%(len(df_anno), len(df_anno_train), len(df_anno_val)))
    return df_img

def get_args():
    parser = argparse.ArgumentParser(description='Json Merge')

    # parameters
    parser.add_argument('--json_all_path', type=str,
                        help='json path to split')
    parser.add_argument('--json_train_path', type=str,
                        help='json path to save the split result -- train part')
    parser.add_argument('--json_val_path', type=str,
                        help='json path to save the split result -- val part')
    parser.add_argument('--val_split_rate', type=float, default=0.1,
                        help='val image number rate in total image, default is 0.1; if val_split_num is set, val_split_rate will not work')
    parser.add_argument('--val_split_num', type=int, default=None,
                        help='val image number in total image, default is None; if val_split_num is set, val_split_rate will not work')
    parser.add_argument('--keep_val_inTrain', type=bool, default=False,
                        help='if true, val part will be in train as well; which means that the content of json_train_path is the same as the content of json_all_path')
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=True,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100,'-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args

if __name__ == '__main__':
    # args = get_args()
    # js_split(args.json_all_path,args.json_train_path,args.json_val_path, args.val_split_rate,  args.val_split_num, args.keep_val_inTrain)

    js_split(js_all_path=r'/data/huilin/data/isds/bd_data/data389_seg/instance_all.json',
             js_train_path=r'/data/huilin/data/isds/bd_data/data389_seg/instance_train.json',
             js_val_path=r'/data/huilin/data/isds/bd_data/data389_seg/instance_val.json',
             val_split_rate=0.9)
