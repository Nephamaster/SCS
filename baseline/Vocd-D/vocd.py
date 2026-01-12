import json
import numpy as np
from lexicalrichness import LexicalRichness

# 数据路径
data_path = r"/mnt/disk4t/huangyishuo/diversity/data/Llama31_SFT_10000_repr.json"

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


vocd_values = []

for idx, sample in enumerate(dataset):
    text = flatten_sample(sample)

    # 基本清洗
    if not text:
        continue

    tokens = text.split()

    # vocd 对极短文本不稳定，通常建议至少 50 token
    if len(tokens) < 50:
        continue

    try:
        lex = LexicalRichness(text)
        vocd = lex.vocd()
        vocd_values.append(vocd)
    except Exception as e:
        # 极少数情况下 vocd 拟合失败，直接跳过
        print(f"[Warning] vocd failed at sample {idx}: {e}")
        continue


# 数据集统计
vocd_values = np.array(vocd_values)

print(f"Valid samples used for vocd: {len(vocd_values)}")

print(f"Average vocd-D : {vocd_values.mean():.4f}")
print(f"Std vocd-D     : {vocd_values.std():.4f}")
print(f"Median vocd-D  : {np.median(vocd_values):.4f}")
