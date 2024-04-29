import json
import os


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
    # coco_eval(anno_json=r'E:\data\2023_defect\yolo_fomat_c5\yolo_fomat_c5\instance_val.json',
    #           pred_json=r'E:\data\2023_defect\yolo_fomat_c5\pred\yolov9-classc5-44\yolov9-classc5-44\best_predictions.json')

    # import timm
    # # for model in timm.list_models():
    # #     print(model)
    # model = timm.create_model('seresnet34')
    # print(model)
    import numpy as np
    from skimage import io
    img_dir = r'E:\img_huilin'
    img_dir2 = r'E:\img_huilin2'
    for file_name in os.listdir(img_dir):
        if not file_name.endswith('.jpg'):
            continue
        img = io.imread(os.path.join(img_dir, file_name))
        img_new = np.zeros_like(img)
        img_new[:, :1920] = img[:, 1920:]
        img_new[:, 1920:] = img[:, :1920]
        io.imsave(os.path.join(img_dir2, file_name),img_new)