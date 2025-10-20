import requests
import os
from typing import Dict, Optional
import time
from tqdm import tqdm

API_KEY = '36c5d60c-676f-401a-99bd-ef8878ceb6fb'
BASE_URL = "http://ec2-54-46-0-164.ap-east-1.compute.amazonaws.com/"

class DataUploader:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL):
        """
        初始化上传器
        :param api_key
        :param base_url
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'X-API-Key': api_key
        }



    def upload_wildcard(self, zip_file_path: str) -> Optional[Dict]:
        """

        :param zip_file_path:
        :return:
        """
        if not os.path.isfile(zip_file_path):
            print(f"error: file not exists! {zip_file_path}")
            return None

        file_size = os.path.getsize(zip_file_path)
        if file_size > 4 * 1024 * 1024 * 1024:  # 4GB in bytes
            print(f"Error: size of files error ({file_size} bytes)")
            return None

        url = f"{self.base_url}/fs/api/v2/data-catalog/wildcard-upload"

        try:
            t1 = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] uploading start")
            with open(zip_file_path, 'rb') as f:
                files = {'file': (os.path.basename(zip_file_path), f)}
                print(f'post info:\nurl:{url}\nheaders:{self.headers}\nfiles:{files}')
                response = requests.post(
                    url,
                    headers=self.headers,
                    files=files
                )
                print(f"respon: {response.text}")
                if response.status_code == 200:
                    print(f"success with {response.status_code}")
                elif response.status_code == 401:
                    print(f"error with {response.status_code}: API error")
                elif response.status_code == 500:
                    print(f"error with {response.status_code}: server error")
                else:
                    print(f"error with {response.status_code}: unknown error")

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] uploading end")
            t2 = time.time()
            print(f"uploading time: {t2 - t1}")
        except Exception as e:
            print(f"unexpected error: {str(e)}")
            return None

    def upload_batch_images(self, zip_file_path: str) -> Optional[Dict]:
        """

        :param zip_file_path:
        :return:
        """
        if not os.path.isfile(zip_file_path):
            print(f"error: file not exists! {zip_file_path}")
            return None

        file_size = os.path.getsize(zip_file_path)
        if file_size > 4 * 1024 * 1024 * 1024:  # 4GB in bytes
            print(f"Error: size of files error ({file_size} bytes)")
            return None

        url = f"{self.base_url}/fs/api/v2/data-catalog/bulk-upload"

        try:
            t1 = time.time()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] uploading start")
            with open(zip_file_path, 'rb') as f:
                files = {'file': (os.path.basename(zip_file_path), f)}
                print(f'post info:\nurl:{url}\nheaders:{self.headers}\nfiles:{files}')
                response = requests.post(
                    url,
                    headers=self.headers,
                    files=files
                )
                print(f"respon: {response.text}")
                if response.status_code == 200:
                    print(f"success with {response.status_code}")
                elif response.status_code == 401:
                    print(f"error with {response.status_code}: API error")
                elif response.status_code == 500:
                    print(f"error with {response.status_code}: server error")
                else:
                    print(f"error with {response.status_code}: unknown error")

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] uploading end")
            t2 = time.time()
            print(f"uploading time: {t2-t1}")
        except Exception as e:
            print(f"unexpected error: {str(e)}")
            return None

    def check_uploaded_batch_images(self):
        url = f"{self.base_url}/emes/api/v2/updateDataCatalogStatus"
        # url = f"{self.base_url}/fs/api/v2/nav/checkBulkUploadImageZip"
        body = {"status": "Upload Complete"}
        print(f'post info:\nurl:{url}\nheaders:{self.headers}\nparams:{body}')
        response = requests.post(
            url,
            headers=self.headers,
            params=body,
        )

        if response.status_code == 200:
            print(f"success with {response.status_code}")
            return response.json()
        elif response.status_code == 401:
            print(f"error with {response.status_code}: API error")
        elif response.status_code == 500:
            print(f"error with {response.status_code}: server error")
        else:
            print(f"error with {response.status_code}: unknown error")
        print(f"respon: {response.text}")

    def check_uploaded_wildcard(self):
        url = f"{self.base_url}/fs/api/v2/nav/checkWildCardUpload"
        # url = f"{self.base_url}/fs/api/v2/nav/checkBulkUploadImageZip"

        params = {"status": "Upload Complete",
                  "dataCatalogId": 606}
        print(f'post info:\nurl:{url}\nheaders:{self.headers}\n'
              f'params:{params}'
              )
        response = requests.post(
            url,
            headers=self.headers,
            params=params,
        )

        if response.status_code == 200:
            print(f"success with {response.status_code}")
            return response.json()
        elif response.status_code == 401:
            print(f"error with {response.status_code}: API error")
        elif response.status_code == 500:
            print(f"error with {response.status_code}: server error")
        else:
            print(f"error with {response.status_code}: unknown error")
        print(f"respon: {response.text}")
    def download_result(self, save_path):
        url = f"{self.base_url}/emes/api/v2/workflow-result/exportResults"
        body = {"subProjectId": 196}
        print(f'post info:'
              f'\nurl:{url}'
              f'\nparams:{body}')
        response = requests.get(
            url,
            params=body,
        )
        total_size = int(response.headers.get('content-length', 0))

        if response.status_code == 200:
            print(f"success with {response.status_code}")
            with open(save_path, 'wb') as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        elif response.status_code == 401:
            print(f"error with {response.status_code}: API error")
        elif response.status_code == 500:
            print(f"error with {response.status_code}: server error")
        else:
            print(f"error with {response.status_code}: unknown error")
        print(f"respon: {response.text}")

    def download_result_only(self):
        url = f"{self.base_url}/emes/api/v2/workflow-result/exportOutputResults"
        body = {
            "subProjectId": 196,
            "queryFrom": 20241013012137699,
            "queryUntil": 20241013012147699,
        }
        print(f'post info:'
              f'\nurl:{url}'
              f'\nparams:{body}')
        response = requests.get(
            url,
            params=body,
        )

        if response.status_code == 200:
            print(f"success with {response.status_code}")

        elif response.status_code == 401:
            print(f"error with {response.status_code}: API error")
        elif response.status_code == 500:
            print(f"error with {response.status_code}: server error")
        else:
            print(f"error with {response.status_code}: unknown error")
        print(f"respon: {response.text}")

    def get_subproject_metadata(self):
        url = f"{self.base_url}/emes/api/v2/getSubprojectMetaData"
        body = {
            "subProjectId": 129,
        }
        print(f'post info:'
              f'\nurl:{url}'
              f'\nparams:{body}')
        response = requests.get(
            url,
            params=body,
        )

        if response.status_code == 200:
            print(f"success with {response.status_code}")
            return response.json()
        elif response.status_code == 401:
            print(f"error with {response.status_code}: API error")
        elif response.status_code == 500:
            print(f"error with {response.status_code}: server error")
        else:
            print(f"error with {response.status_code}: unknown error")
        print(f"get_subproject_metadata respon: {response.text}")

    def update_catalog_status(self,query_status=["Upload Complete"]):
        url = f"{self.base_url}/emes/api/v2/updateDataCatalogStatus"
        params = {"status": query_status}
        response = requests.post(url,headers=self.headers,params=params)
        if response.status_code == 200:
            print("Upload status checked.")
            print(response.json())
        else:
            print(f"Failed to check: {response.status_code}\n{response.text}")

# 使用示例
if __name__ == "__main__":
    # 配置信息
    # API_KEY = '36c5d60c-676f-401a-99bd-ef8878ceb6fb'
    # BASE_URL = "http://ec2-54-46-0-164.ap-east-1.compute.amazonaws.com/"

    # uploader = DataUploader(API_KEY, BASE_URL)

    # zip_path = r"E:\data\202502_signboard\data_annotation\task\task0528\cdu_test\demo_images_10_rename.zip"
    # zip_path = r"E:\data\202502_signboard\PS\20250618\data0618_cam1.zip"
    zip_path = r'E:\data\202502_signboard\PS\20250702\cam_DA4930148.zip'
    print("\nimages uploading...")
    # uploader.upload_batch_images(zip_path)
    # uploader.upload_wildcard(txt_path)
    # uploader.check_uploaded_wildcard()
    print("\ncheck image upload result...")
    uploader.check_uploaded_batch_images()
    # uploader.download_result_only()
    # uploader.download_result('result.zip')

    # http://ec2-54-46-0-164.ap-east-1.compute.amazonaws.com/emes/api/v2/workflow-result/exportResults?subProjectId=205
    # http://ec2-54-46-0-164.ap-east-1.compute.amazonaws.com/emes/api/v2/workflow-result/exportResults?subProjectId=208
    # http://ec2-54-46-0-164.ap-east-1.compute.amazonaws.com/emes/api/v2/workflow-result/exportOutputResults?subProjectId=208