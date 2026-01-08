import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


class Extractor:
    def __init__(self, generator:str="meta-llama/Llama-3.1-8B", embedder:str="meta-llama/Llama-3.1-8B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(generator, trust_remote_code=True)
        if 'llama' in generator.lower():
            tokenizer.pad_token = tokenizer.eos_token
        self.gen_tokenizer = tokenizer
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator, device_map="auto", dtype=torch.float16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(embedder, trust_remote_code=True)
        if 'llama' in embedder.lower():
            tokenizer.pad_token = tokenizer.eos_token
        self.emb_tokenizer = tokenizer
        self.emb_model = AutoModel.from_pretrained(
            embedder, device_map="auto", dtype=torch.float16, trust_remote_code=True)
        self.emb_model.eval()

    def cal_gen_prob(self, sentence:str, max_len:int=4096) -> float:
        """计算输入文本的生成概率"""
        self.gen_tokenizer.padding_side = "right"
        inputs = self.gen_tokenizer(
            sentence, 
            padding=True, 
            truncation=True,
            max_length = max_len, 
            return_tensors='pt'
        ).to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = input_ids.clone()
        with torch.inference_mode():
            outputs = self.gen_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits # [1, seq_len, vocab_size]
        torch.cuda.empty_cache()
        shift_logits = logits[:, :-1, :]  # [1, seq_len-1, vocab_size]
        shift_labels = labels[:, 1:]      # [1, seq_len-1]
        # print('output length: ', outputs.logits.size(1))
        log_softmax = F.log_softmax(shift_logits, dim=-1)
        log_likes = log_softmax.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]
        log_likes_norm = log_likes.mean().item()
        return log_likes_norm
    
    def get_embedding(self, sentence:str, max_len:int=256) -> np.ndarray:
        """获取输入文本的语义向量"""
        encoded_input = self.emb_tokenizer(
            [sentence],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
            return_attention_mask=True
        ).to(self.device)
        
        with torch.inference_mode():
            model_output = self.emb_model(**encoded_input)
            hidden_states = model_output.last_hidden_state  # [1, seq_len, hidden_size]
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除零
            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        torch.cuda.empty_cache()
        return sentence_embeddings.cpu().numpy()[0]