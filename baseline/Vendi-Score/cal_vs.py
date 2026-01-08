import json
from vendi_score import text_utils


def cal_vs(dataset_name, model_path):
    with open(f'/mnt/disk4t/heyuxuan/work/sce/data/All/{dataset_name}.json', 'r') as f:
        data = json.load(f)
    sents = [term['doc'] for term in data]
    vendi = text_utils.embedding_vendi_score(sents, model_path=model_path, device='cuda')
    return vendi


dataset_list = ['Llama31_SFT_10000_Kmeans', 'Llama31_SFT_10000_Kcenter', 'Llama31_SFT_10000_Repr', 'SFT_10000_Random']

for dataset_name in dataset_list:
    vendi = cal_vs(dataset_name, '/mnt/disk4t/heyuxuan/data/models/Qwen/Qwen3-Embedding-8B/')
    print(f'{dataset_name}: {vendi}')