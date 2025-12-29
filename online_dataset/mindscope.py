from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-225e0920-b043-4179-8c16-4c0b32f297c6'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'huilin16'
dataset_name = 'coco_ultralytics'

# api.upload_folder(
#     repo_id=f"{owner_name}/{dataset_name}",
#     folder_path='/path/to/local/dir',
#     commit_message='upload dataset folder to repo',
#     repo_type = 'dataset'
# )

api.upload_file(
    path_or_fileobj=r'\\158.132.186.40\isds\huilin\coco\coco.zip',
    path_in_repo='coco.zip',
    repo_id=f"{owner_name}/{dataset_name}",
    repo_type = 'dataset',
    commit_message='upload dataset file to repo',
)