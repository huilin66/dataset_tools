import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import itertools
from terminaltables import AsciiTable

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--anno_json', type=str, default='/home/val.json', help='training model path')
#     parser.add_argument('--pred_json', type=str, default='', help='data yaml path')
#
#     return parser.parse_known_args()[0]

def pred_convert(anno_json, pred_json, tp_json):
    with open(anno_json) as f:
        data = json.load(f)
    name2id_dict = {}
    img_data = data['images']
    for img_record in img_data:
        name2id_dict[img_record['file_name'].replace('.jpg', '')] = img_record['id']

    with open(pred_json) as f:
        data = json.load(f)
        for i in range(len(data)):
            key = data[i]['image_id']
            data[i]['image_id'] = name2id_dict[key]
        with open(tp_json, 'w') as f:
            output_json = json.dumps(data)
            f.write(output_json)


def coco_eval(anno_json, pred_json, tp_json='temp.json', classwise=False):
    pred_convert(anno_json, pred_json, tp_json)
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(tp_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    if classwise:
        precisions = eval.eval['precision']
        cat_ids = anno.getCatIds()
        results_per_category = []
        for idx, catId in enumerate(cat_ids):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = anno.loadCats(catId)[0]
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            if precision.size:
                ap = np.mean(precision)
            else:
                ap = float('nan')
            results_per_category.append(
                (str(nm["name"]), '{:0.3f}'.format(float(ap))))
            pr_array = precisions[0, :, idx, 0, 2]
            recall_array = np.arange(0.0, 1.01, 0.01)


        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP'] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print('Per-category of {} AP: \n{}'.format('bbox', table.table))
        print("per-category PR curve has output to {} folder.".format(
            'bbox' + '_pr_curve'))



if __name__ == '__main__':
    pass
    # opt = parse_opt()
    # anno_json = opt.anno_json
    # pred_json = opt.pred_json

    # anno = COCO(anno_json)  # init annotations api
    # pred = anno.loadRes(pred_json)  # init predictions api
    # eval = COCOeval(anno, pred, 'bbox')
    # eval.evaluate()
    # eval.accumulate()
    # eval.summarize()
    # coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\val23\val23\_annotations.coco.json',
    #           pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\val23\val23\predictions.json')
    # coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json',
    #           pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\yolov9-classc5-44\yolov9-classc5-44\best_predictions.json')
    # coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json',
    #           pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\yolov9-classc5-45\yolov9-classc5-45\best_predictions.json')

    coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json',
              pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\yolov9-classc5-56\yolov9-classc5-56\best_predictions.json',
              classwise=True)

