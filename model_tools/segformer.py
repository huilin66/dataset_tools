from transformers import SegformerForSemanticSegmentation
from PIL import Image
import requests


model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

print(model)
