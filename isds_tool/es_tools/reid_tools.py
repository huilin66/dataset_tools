import os
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
import pandas as pd

class FastReID_ONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=[
            # "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]
        return img

    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def extract(self, patch):
        input_tensor = self.preprocess(patch)
        feat = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0]
        feat = self.normalize(feat, axis=1)
        return feat

def _cosine_sim(a, b):
    return np.dot(a, b.T)


def imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)

def images2feature(input_dir, output_dir, model_path):
    reid_model = FastReID_ONNX(model_path)
    os.makedirs(output_dir, exist_ok=True)

    img_list = os.listdir(input_dir)
    for img_name in tqdm(img_list):
        img_path = os.path.join(input_dir, img_name)
        img = imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = reid_model.extract(img)
        save_name = Path(img_name).stem + ".npy"
        np.save(os.path.join(output_dir, save_name), embedding)


def get_sim(input_dir, search_dir, save_path=None):
    if save_path is None:
        save_path = input_dir + '.csv'
    input_list = os.listdir(input_dir)
    search_list = os.listdir(search_dir)

    input_embedding = [np.load(os.path.join(input_dir, input_name)) for input_name in tqdm(input_list)]
    search_embedding = [np.load(os.path.join(search_dir, search_name)) for search_name in tqdm(search_list)]

    input_embedding = np.array(input_embedding).reshape(len(input_embedding), -1)
    search_embedding = np.array(search_embedding).reshape(len(search_embedding), -1)
    print(input_embedding.shape, search_embedding.shape)
    sims = _cosine_sim(input_embedding, search_embedding)

    df = pd.DataFrame(sims, columns=search_list, index=input_list)
    print(df.shape)
    df.to_csv(save_path, encoding='utf-8-sig')
