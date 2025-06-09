import os
import datetime
import shutil
import decimal
from pathlib import Path
from tqdm import tqdm
PREFIX = 'test_'
TIMESTAMP_GAP_NUM = 158116921

def name_swift(src_name, prefix, timestamp_gap_num):
    src_timestamp = float(Path(src_name).stem)
    unix_timestamp = decimal.Decimal(src_timestamp)
    added_seconds = decimal.Decimal(timestamp_gap_num)
    dst_timestamp = unix_timestamp + added_seconds
    dt = datetime.datetime.utcfromtimestamp(dst_timestamp)
    milliseconds = int((dst_timestamp - int(dst_timestamp)) * 1000)
    formatted_time = dt.strftime("%Y%m%d%H%M%S") + f"{milliseconds:03d}"
    dst_name = f"{prefix}{formatted_time}{Path(src_name).suffix}"
    return dst_name

def image_name_swift(src_dir, dst_dir, prefix=PREFIX, timestamp_gap_num=TIMESTAMP_GAP_NUM):
    os.makedirs(dst_dir, exist_ok=True)
    image_list = os.listdir(src_dir)
    for image_name in tqdm(image_list):
        image_name_dst = name_swift(image_name, prefix, timestamp_gap_num)
        src_path = os.path.join(src_dir, image_name)
        dst_path = os.path.join(dst_dir, image_name_dst)
        shutil.copyfile(src_path, dst_path)
        # print(dst_path)

if __name__ == '__main__':
    pass
    src_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\cdu_test\demo_images_1000'
    dst_dir = r'E:\data\202502_signboard\data_annotation\task\task0528\cdu_test\demo_images_1000_rename'
    image_name_swift(src_dir, dst_dir)