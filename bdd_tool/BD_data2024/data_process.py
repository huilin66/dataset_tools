import os
import shutil
import pandas as pd
from tqdm import tqdm
from config import *


def data_merge(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    df = pd.DataFrame(None, columns=['image_name', 'src_path'])
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith('.JPG'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.basename(input_path))
                shutil.copy(input_path, output_path)
                df.loc[len(df)] = [os.path.basename(input_path), os.path.dirname(input_path)]
            else:
                print(file)
    csv_path = os.path.join(input_dir, 'summary.csv')
    df.to_csv(csv_path)
    print(df)


if __name__ == '__main__':
    pass
    data_merge(ROOT_DIR, MERGE_DIR)
