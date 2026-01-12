import os
import json
import numpy as np
from lexicalrichness import LexicalRichness

# -----------------------------
# 数据集配置
# -----------------------------
BASE_DIR = "/mnt/disk4t/heyuxuan/work/sce/data/All"
DATASETS = {
    "Qwen3_kmeans": "Qwen3_kmeans.json",
    "Qwen3_kcenter": "Qwen3_kcenter.json",
    "Qwen3_repr": "Qwen3_repr.json",
    "Qwen3_random": "Qwen3_random.json",
    "Llama31_kmeans": "Llama31_kmeans.json",
    "Llama31_kcenter": "Llama31_kcenter.json",
    "Llama31_repr": "Llama31_repr.json",
    "Llama31_random": "Llama31_random.json"
}

# -----------------------------
# 文本提取函数
# -----------------------------
def extract_doc(sample):
    """
    仅提取每个样本中的 doc 字段
    """
    text = sample.get("doc", "")
    if not isinstance(text, str):
        return ""
    return text.strip()

# -----------------------------
# vocd 计算函数（单数据集）
# -----------------------------
def compute_vocd(dataset):
    vocd_values = []

    for idx, sample in enumerate(dataset):
        text = extract_doc(sample)

        # 基本清洗
        if not text:
            continue

        tokens = text.split()
        if len(tokens) < 50:  # vocd 对极短文本不稳定
            continue

        try:
            lex = LexicalRichness(text)
            vocd_values.append(lex.vocd())
        except Exception as e:
            # print(f"[Warning] vocd failed at sample {idx}: {e}")
            continue

    return np.array(vocd_values)

# -----------------------------
# 批量处理所有数据集
# -----------------------------
for name, filename in DATASETS.items():
    path = os.path.join(BASE_DIR, filename)

    # 读取数据
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"\nProcessing dataset: {name}")
    print(f"Total samples: {len(dataset)}")

    vocd_values = compute_vocd(dataset)

    if len(vocd_values) == 0:
        print("No valid samples for vocd calculation!")
        continue

    # 输出统计指标
    print(f"Valid samples used for vocd: {len(vocd_values)}")
    print(f"Average vocd-D : {vocd_values.mean():.4f}")
    print(f"Std vocd-D     : {vocd_values.std():.4f}")
    print(f"Median vocd-D  : {np.median(vocd_values):.4f}")
