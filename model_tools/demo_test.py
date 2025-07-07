from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()

inputs = tokenizer("This is a test sentence", return_tensors="pt").to('cuda')
outputs = model(**inputs)
print(outputs.logits)