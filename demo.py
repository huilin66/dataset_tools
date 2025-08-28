# # 首次使用需要安装aistudio-sdk库
# # pip install --upgrade aistudio-sdk
#
# import os
# # 需要填写aistudio-access-token, 在我的控制台--令牌获取
# os.environ["AISTUDIO_ACCESS_TOKEN"] = "91105a6b209316f8a84664af5d6d64f35781faeb"
#
# #上传单个文件
# from aistudio_sdk.hub import upload
# res = upload(
#     # 填写数据集详情页面中的repo_id
#     repo_id='huilin/ps_data3',
#     # 填写要上传的文件在本地的路径，如'./path/to/local/README.md'
#     path_or_fileobj=r'Y:\ZHL\isds\PS\task0725\ymt-2\track_data\cam_DA4930148_img.zip',
#     # 填写上传至repo后的文件路径及文件名，如填写'README.md'，则会在master分支的根目录内，上传README.md
#     path_in_repo='cam_DA4930148_image.zip',
#     # 填写commit信息，非必填
#     commit_message='upload dataset file to repo',
#     # 填写仓库类型为dataset，上传数据集文件时为必填项
#     repo_type = 'dataset'
# )
# print(res)


# 首次使用需要安装aistudio-sdk库
# pip install --upgrade aistudio-sdk

import os
# 如下载私密数据集，需要填写数据集所有者的aistudio-access-token, 在我的控制台--令牌处获取
os.environ["AISTUDIO_ACCESS_TOKEN"] = "677190c4aa394ff306562148b74580cc1a329003"
from aistudio_sdk.snapshot_download import snapshot_download

# 首次使用需要安装aistudio-sdk库
# pip install --upgrade aistudio-sdk


res = snapshot_download(
    # 填写数据集详情页面中的repo_id，如myname/myreponame
    repo_id='huilin/ps_data3',
    # 填写分支版本，如master
    revision='master',
    # 填写本地保存路径，如当前文件夹'./'
    local_dir='./',
    # 填写仓库类型为dataset，下载数据集文件时为必填项
    repo_type='dataset'
)
print(res)