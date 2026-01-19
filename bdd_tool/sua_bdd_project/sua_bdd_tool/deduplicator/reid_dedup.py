import os
import cv2
import numpy as np
import onnxruntime as ort


class FastReID_ONNX:
    def __init__(self, model_path, device_id=0):
        providers = [("CUDAExecutionProvider", {"device_id": device_id})]
        if model_path and os.path.exists(model_path):
            print(f"🧠 Loading ReID Model from {model_path}...")
            self.session = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(), providers=providers)
        else:
            raise ValueError(f"{model_path} ReID model path is invalid or does not exist.")
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, img):
        # Resize to (128, 256) as required by FastReID
        img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]
        return img

    def normalize(self, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def extract(self, patch):
        input_tensor = self.preprocess(patch)
        feat = self.session.run(None, {self.input_name: input_tensor})[0]
        feat = self.normalize(feat, axis=1)
        return feat.flatten() # Return 1D array for easier storage



def compute_cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    if emb1 is None or emb2 is None:
        return 0.0
    return np.dot(emb1, emb2)