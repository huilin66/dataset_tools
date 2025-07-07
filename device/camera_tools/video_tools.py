import os
import cv2
import pandas as pd
from tqdm import tqdm
FRAME_GAP = 10

def video_sta(video_dir):
    file_list = os.listdir(video_dir)
    frame_counts = 0
    for file_name in file_list:
        if not file_name.endswith('.avi'):
            continue

        # 打开视频文件
        video_path = os.path.join(video_dir, file_name)
        video = cv2.VideoCapture(video_path)

        # 获取视频基本信息
        fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高
        duration = frame_count / fps  # 视频时长（秒）
        frame_counts += frame_count

        # 打印信息
        print(video_path)
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"Frame size: {frame_width}x{frame_height}")
        print(f"Duration: {duration} seconds")
        print()
        video.release()
    print(f"Total frames: {frame_counts}")

def video2frame(video_dir, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    file_list = os.listdir(video_dir)
    for file_name in file_list:
        if not file_name.endswith('.avi'):
            continue
        video_path = os.path.join(video_dir, file_name)
        video = cv2.VideoCapture(video_path)
        print('processing', video_path)
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % FRAME_GAP == 0:
                frame_path = os.path.join(frame_dir, file_name.replace('.avi', '_%03d.png'%frame_count))
                cv2.imwrite(frame_path, frame)

            frame_count += 1

def label_process(input_dir, output_dir, frame_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        if not file_name.endswith('.csv'):
            continue
        file_path = os.path.join(input_dir, file_name)
        df = pd.read_csv(file_path, header=None, index_col=None, names=['frame', 'object', 'lt_x', 'lt_y', 'w', 'h', 'category', 'cat1', 'cat2', 'cat3'])

        frame_nums = df['frame'].unique().tolist()
        for frame_num in frame_nums:
            frame_name = file_name.replace('-gt.csv', '_%03d.png'%frame_num)
            frame_path = os.path.join(frame_dir, frame_name)
            label_path = os.path.join(output_dir, frame_name.replace('.png', '.txt'))
            if os.path.exists(frame_path):
                frame_height, frame_width = cv2.imread(frame_path).shape[:2]
                df_frame = df[df['frame'] == frame_num].copy()
                df_frame['category'] += 1
                df_frame['center_x'] = df_frame['lt_x'] + df_frame['w']*0.5
                df_frame['center_y'] = df_frame['lt_y'] + df_frame['h']*0.5
                df_frame['center_x'] /= frame_width
                df_frame['center_y'] /= frame_height
                df_frame['w'] /= frame_width
                df_frame['h'] /= frame_height
                df_frame_det = df_frame[['category', 'center_x', 'center_y', 'w', 'h']]
                df_frame_det = df_frame_det[df_frame_det['center_x'] <= 1]
                df_frame_det = df_frame_det[df_frame_det['center_y'] <= 1]
                df_frame_det.to_csv(label_path, index=False, header=False, sep=' ')


        # count = df[df['category'] != -1].shape[0]
        # count1 = df[df['cat1'] != -1].shape[0]
        # count2 = df[df['cat2'] != -1].shape[0]
        # count3 = df[df['cat3'] != -1].shape[0]
        #
        # print(file_name, count, count1, count2, count3)

if __name__ == '__main__':
    pass
    # video_sta(video_dir=r'E:\data\tp\sar_det\train1')
    # video_sta(video_dir=r'E:\data\tp\sar_det\TestA')
    # video2frame(
    #     video_dir=r'E:\data\tp\sar_det\train1',
    #     frame_dir=r'E:\data\tp\sar_det\images',
    # )
    label_process(
        input_dir=r'E:\data\tp\sar_det\train1',
        output_dir=r'E:\data\tp\sar_det\labels',
        frame_dir=r'E:\data\tp\sar_det\images',
    )