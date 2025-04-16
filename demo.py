from datasets import load_dataset
from tqdm import tqdm

culturalvqa_dataset = load_dataset('mair-lab/CulturalVQA')

for data in tqdm(culturalvqa_dataset['test']):
    # print(data)
    image, question, uid = data['image'], data['question'], data['u_id']

    if not isinstance(question, str):
        print(image, question, uid)