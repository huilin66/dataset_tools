import json


def coco_eval(anno_json, pred_json):
    # label_file = r'E:\data\2023_defect\yolo_fomat_c5\pred\val23\val23\_annotations.coco.json'
    with open(anno_json) as f:
        data = json.load(f)
        print(len(data["images"]))


    # label_file = r'E:\data\2023_defect\yolo_fomat_c5\pred\val23\val23\predictions.json'
    with open(pred_json) as f:
        data = json.load(f)
        print(len(data))
        with open(pred_json, 'w') as f:
            output_json = json.dumps(data)
            f.write(output_json)

if __name__ == '__main__':
    pass
    coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json',
              pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\yolov9-classc5-44\yolov9-classc5-44\best_predictions.json')