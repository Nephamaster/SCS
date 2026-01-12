import json
import random
import numpy as np
import os

random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. 数据集配置
# -----------------------------
BASE_DIR = r"/mnt/disk4t/heyuxuan/work/sce/data/All"

DATASETS = {
    "Qwen3_kmeans":    "Qwen3_kmeans_2.json",
    "Qwen3_kcenter":   "Qwen3_kcenter_2.json",
    "Qwen3_repr":      "Qwen3_repr_2.json",
    "Qwen3_random":    "Qwen3_random_2.json",
    "Llama31_kmeans":  "Llama31_kmeans_2.json",
    "Llama31_kcenter": "Llama31_kcenter_2.json",
    "Llama31_repr":    "Llama31_repr_2.json",
    "Llama31_random":  "Llama31_random_2.json"
}

# -----------------------------
# 2. 文本提取函数
# -----------------------------
def extract_doc(sample):
    """
    仅提取每个样本中的 doc 字段作为文本
    """
    text = sample.get("doc", "")
    if not isinstance(text, str):
        return ""
    return text.strip()


# -----------------------------
# 3. TTR 抽样参数
# -----------------------------
sample_size = 30  # 每条样本抽取 30 个 token
num_draws = 10    # 每条样本重复抽样次数

# -----------------------------
# 4. 批量处理数据集
# -----------------------------
for dataset_name, file_name in DATASETS.items():
    file_path = os.path.join(BASE_DIR, file_name)
    
    # 读取 JSON
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"\n=== Processing dataset: {dataset_name} ===")
    print(f"Total samples: {len(dataset)}")

    sample_ttrs = []

    for sample in dataset:
        text = extract_doc(sample)
        tokens = text.split()

        if len(tokens) == 0:
            continue  # 跳过空文本

        current_sample_size = min(sample_size, len(tokens))

        ttr_draws = []
        for _ in range(num_draws):
            sub_tokens = random.sample(tokens, current_sample_size)
            ttr = len(set(sub_tokens)) / current_sample_size
            ttr_draws.append(ttr)

        sample_ttrs.append(np.mean(ttr_draws))

    if len(sample_ttrs) == 0:
        print("Warning: No valid samples for TTR computation.")
        continue

    avg_ttr = np.mean(sample_ttrs)
    print(f"Sample-based Average TTR (random 30-token sampling): {avg_ttr:.4f}")
