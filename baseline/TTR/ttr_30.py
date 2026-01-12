import json
import random
import numpy as np


random.seed(42)
np.random.seed(42)


# 数据路径
data_path = r"/mnt/disk4t/heyuxuan/work/sce/data/All/Qwen3_kcenter_2.json"

# 读取数据
with open(data_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total samples: {len(dataset)}")

def flatten_sample(sample):
    """
    将一个 sample 的所有 conversations 拼接成一段文本
    """
    texts = [turn.get("value", "") for turn in sample.get("conversations", [])]
    return " ".join(texts).strip()

# 配置抽样参数
sample_size = 30  # 每条样本抽取 30 个 token
num_draws = 10    # 每条样本重复抽样次数

sample_ttrs = []

for sample in dataset:
    text = flatten_sample(sample)
    tokens = text.split()
    
    if len(tokens) == 0:
        continue  # 跳过空文本
    
    # 如果文本长度 < sample_size，则直接用全部 tokens
    current_sample_size = min(sample_size, len(tokens))
    
    ttr_draws = []
    for _ in range(num_draws):
        sub_tokens = random.sample(tokens, current_sample_size)
        ttr = len(set(sub_tokens)) / current_sample_size
        ttr_draws.append(ttr)
    
    # 样本 TTR = 抽样平均
    sample_ttrs.append(np.mean(ttr_draws))

# 数据集平均 TTR
avg_ttr = np.mean(sample_ttrs)
print(f"Sample-based Average TTR (random 30-token sampling): {avg_ttr:.4f}")
