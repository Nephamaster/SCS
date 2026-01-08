import json
from codes.utils.ent import *


def cal_ent(dataset_name):
    with open(f'data/{dataset_name}.json', 'r') as f:
        data = json.load(f)
    calculator1 = Ent1Calculator()
    calculator2 = Ent2Calculator()
    calculator3 = Ent3Calculator()
    ent1 = calculator1.calculate(data)[0]
    ent2 = calculator2.calculate(data)[0]
    ent3 = calculator3.calculate(data)[0]
    return ent1, ent2, ent3

# dataset_list = ['MMLU_900_MedQA_100_3', 'MMLU_800_MedQA_200_3', 'MMLU_700_MedQA_300_3', 
#                 'MMLU_600_MedQA_400_3', 'MMLU_500_MedQA_500_3', 'MMLU_400_MedQA_600_3', 
#                 'MMLU_300_MedQA_700_3', 'MMLU_200_MedQA_800_3', 'MMLU_100_MedQA_900_3']
dataset_list = ['Llama31_SFT_10000_Kmeans', 'Llama31_SFT_10000_Kcenter', 'Llama31_SFT_10000_Repr', 'SFT_10000_Random']

for dataset_name in dataset_list:
    ent1, ent2, ent3 = cal_ent(dataset_name)
    print(f'[{dataset_name}] k=1: {ent1}, k=2: {ent2}, k=3: {ent3}')
