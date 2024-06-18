# import torch
# weight_path = r'E:\repository\yolov9-seg2\ckpt\yolov9-c-seg.pt'
# # data = torch.load()
# data = torch.load(weight_path, map_location='cpu')
# print(data)

import os
# file_list = os.listdir(r'E:\data\0417_signboard\data0521_m\yolo_rgbtc_correct\offset_down')
# for file in file_list:
#     print('"%s",'%file )

print(os.path.exists(r'E:\data\0111_testdata\demo_data\polyu_t_vis_crop\DJI_0469_JPG.rf.b58289a6aa49f36dc4c7fa97b0fa2f6a\DJI_0469_JPG.rf.b58289a6aa49f36dc4c7fa97b0fa2f6a_220_170_254_190.png'))