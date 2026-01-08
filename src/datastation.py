import os
import json
from transformers import AutoTokenizer


class DatasetAnalyzer:
    def __init__(self, dataset:str, tokenize_model:str='FacebookAI/xlm-roberta-large'):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenize_model,
            clean_up_tokenization_spaces=False,
            model_max_length=4096)
        self.raw_dir = '../data/raw/'
        self.flat_list:list[str] = []
    
    def flatten(self):
        with open(os.path.join(self.raw_dir, self.dataset+'.json'), 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        for data in data_list:
            doc = ''
            for item in data['conversations']:
                doc += item['value']
            self.flat_list.append(doc)
        return self.flat_list
    
    def tokenize(self):
        tokens = [self.tokenizer.tokenize(sentence) for sentence in self.flat_list]
        token_nums = [len(token) for token in tokens]
        self.avg_token_num = avg_token_num = sum(token_nums) / len(self.flat_list) if len(self.flat_list) > 0 else 0
        return tokens, token_nums, avg_token_num


