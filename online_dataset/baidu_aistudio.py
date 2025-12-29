# 首次使用需要安装aistudio-sdk库
# pip install --upgrade aistudio-sdk

import os
# 需要填写aistudio-access-token, 在我的控制台--令牌获取
os.environ["AISTUDIO_ACCESS_TOKEN"] = "91105a6b209316f8a84664af5d6d64f35781faeb"

#上传单个文件
from aistudio_sdk.hub import upload_file
res = upload_file(
    # 填写数据集详情页面中的repo_id
    repo_id='huilin/coco_ultralytics',
    # 填写要上传的文件在本地的路径，如'./path/to/local/README.md'
    path_or_fileobj=r'\\158.132.186.40\isds\huilin\coco\coco.zip',
    # 填写上传至repo后的文件路径及文件名，如填写'README.md'，则会在master分支的根目录内，上传README.md
    path_in_repo='coco.zip',
    # 填写commit信息，非必填
    commit_message='upload dataset file to repo',
    # 填写仓库类型为dataset，上传数据集文件时为必填项
    repo_type = 'dataset'
)
print(res)
# aistudio upload huilin/coco_ultralytics \\158.132.186.40\isds\huilin\coco\coco.zip coco.zip --repo-type dataset --token 91105a6b209316f8a84664af5d6d64f35781faeb
# aistudio download coco.zip --dataset huilin/coco_ultralytics --local_dir ./dataset/coco

# #上传文件夹
# from aistudio_sdk.hub import upload_folder
# res = upload_folder(
#     # 填写数据集详情页面中的repo_id
#     repo_id='huilin/coco_ultralytics',
#     # 填写要上传的文件在本地的路径，如'./path/to/local/dir'
#     folder_path=r'\\158.132.186.40\isds\huilin\coco\coco\coco',
#     # 填写上传至repo后的文件路径，如填写'data/'，则会将文件上传至data目录内；或不填，则默认上传至master分支的根目录内
#     path_in_repo='data/',
#     # 填写commit信息，非必填
#     commit_message='upload dataset folder to repo',
#     # 填写仓库类型为dataset，上传数据集文件时为必填项
#     repo_type = 'dataset'
# )
# print(res)