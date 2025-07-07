import os
import pandas as pd
from tqdm import tqdm
from skimage import io
from pathlib import Path
input_dir = r'/localnvme/data/added_data/rgbt/Testset/TIR_infer_obj_rgb-[yolov8x]-0.502/labels'
rgb_dir = r'/localnvme/data/added_data/rgbt/Testset/RGB'
tir_dir = r'/localnvme/data/added_data/rgbt/Testset/TIR'
output_path = r'/localnvme/data/added_data/rgbt/submission050t.csv'


def get_result(method, idx, image_ids, cat_ids, bboxs, scores):
    if method == 1:
        if len(bboxs)>1:
            cat_ids = ','.join(map(str, cat_ids))
            bboxs = ','.join(map(str, bboxs))
            scores = ','.join(map(str, scores))
        elif len(bboxs)==1:
            cat_ids = cat_ids[0]
            bboxs = bboxs[0]
            scores = scores[0]
        record = [idx, image_ids, cat_ids, bboxs, scores]
    elif method == 2:
        cat_ids = ','.join(map(str, cat_ids))
        bboxs = ','.join(map(str, bboxs))
        scores = ','.join(map(str, scores))
        record = [idx, image_ids, cat_ids, bboxs, scores]
    elif method == 3:
        cat_ids = ','.join(map(str, cat_ids))
        bboxs = ','.join(map(str, bboxs))
        scores = ','.join(['1' for _ in scores])
        record = [idx, image_ids, cat_ids, bboxs, scores]
    else:
        record = [idx, image_ids, cat_ids, bboxs, scores]
    return record



def result_swift(input_dir, image_dir, output_path, method=1):
    pass
    image_list = os.listdir(image_dir)
    df_all = pd.DataFrame(None, columns=['id', 'image_id', 'category_id', 'bbox', 'score'])
    for idx, image_name in enumerate(tqdm(image_list)):
        image_path = os.path.join(image_dir, image_name)
        pred_path = os.path.join(input_dir, Path(image_name).stem+'.txt')
        image_ids = Path(image_name).stem
        image = io.imread(image_path)
        h, w = image.shape[:2]
        if not os.path.exists(pred_path):
            df_all.loc[len(df_all)] = [idx, image_ids, 0, "[0, 0, 0, 0]", 0]
        else:
            df = pd.read_csv(pred_path, names=['class_id', 'x', 'y', 'w', 'h', 'conf'], sep=' ')
            df['x'] *= w
            df['y'] *= h
            df['w'] *= w
            df['h'] *= h
            cat_ids = df['class_id'].to_list()
            bboxs = df[['x', 'y', 'w', 'h']].values.tolist()
            scores = df['conf'].to_list()
            result = get_result(method, idx, image_ids, cat_ids, bboxs, scores)
            df_all.loc[len(df_all)] = result
    df_all.to_csv(output_path, index=False)

if __name__ == '__main__':
    pass
    result_swift(input_dir, tir_dir, output_path, method=1)

