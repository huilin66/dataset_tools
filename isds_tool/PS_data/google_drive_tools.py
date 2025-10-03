import os
import io
from concurrent.futures import ThreadPoolExecutor
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_with_google(token_path, client_secret_path):
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token_file:
            token_file.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service


def download_large_file(service, file_id, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path) or file_path.endswith('.pcd'):
        print(f"⚠️ 已存在，跳过: {file_path}")
        return
    print(f"⬇️ Downloading {file_path}")
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"⬇️ Downloading {file_path}: {int(status.progress() * 100)}%")
    print(f"✅ Finished: {file_path}")

def download_folder_recursive(service, folder_id, save_path):
    os.makedirs(save_path, exist_ok=True)
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        file_mime = item['mimeType']
        full_path = os.path.join(save_path, file_name)

        if file_mime == 'application/vnd.google-apps.folder':
            download_folder_recursive(service, file_id, full_path)
        else:
            download_large_file(service, file_id, full_path)

def download_subfolder_task(folder_obj, root_save_path, token_path, client_secret_path):
    # 每个线程都单独认证，避免多线程共享service导致问题
    service = authenticate_with_google(token_path, client_secret_path)
    folder_id = folder_obj['id']
    folder_name = folder_obj['name']
    target_path = os.path.join(root_save_path, folder_name)
    print(f"\n📁 Starting folder: {folder_name}")
    download_folder_recursive(service, folder_id, target_path)

def download_all_subfolders_parallel(token_path, client_secret_path, root_folder_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 主线程先获取子文件夹列表
    service = authenticate_with_google(token_path, client_secret_path)
    query = f"'{root_folder_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])

    print(f"将并发下载 {len(folders)} 个子文件夹...\n")

    with ThreadPoolExecutor(max_workers=len(folders)) as executor:
        for folder in folders:
            executor.submit(download_subfolder_task, folder, save_dir, token_path, client_secret_path)

