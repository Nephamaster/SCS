import json
import random


random.seed(123)

with open('../data/All/SFT.json', 'r') as f:
    data = json.load(f)

indices = [i for i in range(len(data))]
random.shuffle(indices)

selected = indices[:10000]

selected_data = [data[i] for i in selected]

with open('../data/All/Llama31_random_2.json', 'w') as f:
    json.dump(selected_data, f, indent=2, ensure_ascii=False)
with open('../data/All/Qwen3_random_2.json', 'w') as f:
    json.dump(selected_data, f, indent=2, ensure_ascii=False)
# with open('../../LLaMA-Factory/data/random.json', 'w') as f:
#     json.dump(selected_data, f, indent=2, ensure_ascii=False)