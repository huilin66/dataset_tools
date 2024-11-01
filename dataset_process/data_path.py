from dataset_merge import *

# img1_path = r'E:\data\2024_defect\2024_defect_pure_yolo_final\cracks-spalls-segmentation-ybu6m\train\images\886_yang_jpg.rf.580d21bbc0c7a10b9a30efaa7145b765.jpg'
# img2_path = r'E:\data\2024_defect\2024_defect_pure_yolo_final\cracks-spalls-segmentation-ybu6m\train\images\887_yang_jpg.rf.25ad77ed1bf7f5e42a05ea131b5eea09.jpg'
# img1 = io.imread(img1_path)
# img2 = io.imread(img2_path)
# # mirror_simple_judge(img1, img2)
# same_judge(img1, img2)
data_dir = r'E:\data\2024_defect\2024_defect_pure_yolo'
get_remained_img(data_dir, 'data.csv')
get_remained_img(data_dir, 'data_rmaug.csv')
get_remained_img(data_dir, 'data_rmaug_rmshift.csv')