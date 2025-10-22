import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def csv_analysis(csv_path):
    df = pd.read_csv(csv_path, header=0, index_col=0)
    return df

def df_col_plot(df):
    plt.figure(figsize=(15, 10))
    sns.boxplot(df)
    plt.tight_layout()
    plt.show()
    plt.close()
    # for col in df.columns:
    #     plt.figure(figsize=(15, 10))
    #     sns.histplot(df[col])
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()

def df_col_sta(df):
    result = {}
    count_sum = 0
    for col in df.columns:
        mask = df[col] > 0.5
        indices = df[mask].index.tolist()
        count = len(indices)
        if count > 100:
            mask = df[col] > 0.6
            indices = df[mask].index.tolist()
            count = len(indices)
            if count > 100:
                mask = df[col] > 0.7
                indices = df[mask].index.tolist()
                count = len(indices)
        count_sum += count
        info = {'count':count, 'indices':indices}
        result[col] = info
        print(info)
    print(count_sum)
    return result


def get_stem2name(input_dir):
    name_list = os.listdir(input_dir)
    stem2name_dict = {Path(name).stem:name for name in name_list}
    return stem2name_dict

def copy_file(result, input_dir, output_dir, ref_dir):
    count_sum, count_copy = 0, 0
    stem2name_dict = get_stem2name(input_dir)
    ref_list = [Path(file_name).stem for file_name in os.listdir(ref_dir)]
    source_img_list = []
    for i, (k, v) in enumerate(result.items()):
        input_stem = Path(k).stem
        output_sub_dir = os.path.join(output_dir, input_stem)
        output_sub_sub_dir = os.path.join(output_sub_dir, 'matched_images')
        shutil.rmtree(output_sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)
        os.makedirs(output_sub_sub_dir, exist_ok=True)

        input_path = os.path.join(input_dir, stem2name_dict[input_stem])
        output_sub_path = os.path.join(output_sub_dir, stem2name_dict[input_stem])
        shutil.copy(input_path, output_sub_path)

        for match_name in tqdm(v['indices'], desc=f'{i}/{len(result)} {input_stem}'):
            count_sum += 1
            match_stem = Path(match_name).stem
            if match_stem == input_stem:
                continue
            match_image_stem = match_stem.rsplit('_', 1)[0]
            if match_image_stem in ref_list:
                continue
            input_path = os.path.join(input_dir, stem2name_dict[match_stem])
            output_sub_sub_path = os.path.join(output_sub_sub_dir, stem2name_dict[match_stem])
            shutil.copy(input_path, output_sub_sub_path)
            count_copy += 1
            source_img_list.append(match_image_stem)
    print(f'find {count_sum}, copy {count_copy}')
    source_img_list = list(set(source_img_list))
    df_source_img = pd.DataFrame(source_img_list, columns=['file_name'])
    df_source_img.to_csv(output_dir+'.csv', index=False)
if __name__ == '__main__':
    csv_path = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1022/result_analysis/images_crop_reid_feature.csv'
    input_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1022/result_analysis/images_crop'
    output_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine/result_analysis/data_copy'
    ref_dir = r'/localnvme/data/added_data/check1021/data29_check1021_mseg_c6_broken_refine/images'
    df = csv_analysis(csv_path)
    result = df_col_sta(df)
    copy_file(result, input_dir, output_dir, ref_dir)