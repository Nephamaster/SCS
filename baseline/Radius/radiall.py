import os
import sqlite3
import pickle
import numpy as np

# -----------------------------
# 1. 读取 embedding
# -----------------------------
def read_feature(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    dataset = db_path.split('/')[-1].replace('.db','')
    cursor.execute(f"SELECT * FROM {dataset}")
    rows = cursor.fetchall()
    features = {}
    for i, row in enumerate(rows):
        embedding = pickle.loads(row[3])
        features[i] = {'embedding': embedding}
    conn.close()
    return features


# =============================
# 新增：多数据集配置（仅新增）
# =============================
BASE_DIR = "/mnt/disk4t/heyuxuan/work/sce/output/feature/"

DATASETS = {
    "Qwen3_kmeans":      "Qwen3_kmeans_2.db",
    "Qwen3_kcenter":     "Qwen3_kcenter_2.db",
    "Qwen3_repr":        "Qwen3_repr_2.db",
    "Qwen3_random":      "Qwen3_random_2.db",
    "Llama31_kmeans":    "Llama31_kmeans_2.db",
    "Llama31_kcenter":   "Llama31_kcenter_2.db",
    "Llama31_repr":      "Llama31_repr_2.db",
    "Llama31_random":    "Llama31_random_2.db",
}


# =============================
# 新增：批量循环（仅新增）
# =============================
for name, db_file in DATASETS.items():

    print("=" * 70)
    print(f"Dataset: {name}")

    db_path = os.path.join(BASE_DIR, db_file)
    features = read_feature(db_path)

    embeddings = np.array(
        [v['embedding'] for v in features.values()],
        dtype=np.float64
    )

    print(f"Total samples loaded: {embeddings.shape[0]}")
    print(f"Original embeddings shape: {embeddings.shape}")

    # -----------------------------
    # 3. 处理 NaN / Inf（原样保留）
    # -----------------------------
    embeddings = np.nan_to_num(
        embeddings,
        nan=1e-10,
        posinf=1e-10,
        neginf=1e-10
    )

    print("Any NaN remaining:", np.isnan(embeddings).any())
    print("Any Inf remaining:", np.isinf(embeddings).any())

    # -----------------------------
    # 4. 计算 Radius（原样保留）
    # -----------------------------
    sigma = np.std(embeddings, axis=0, ddof=1)

    epsilon = 1e-12
    sigma_safe = np.maximum(sigma, epsilon)

    radius = np.exp(np.mean(np.log(sigma_safe)))

    print(f"Radius (geometric mean of per-dim std, stable): {radius:.6f}")

    # -----------------------------
    # 5. 维度级统计（原样保留）
    # -----------------------------
    print(f"Mean per-dim std:   {sigma.mean():.6f}")
    print(f"Std per-dim std:    {sigma.std():.6f}")
    print(f"Median per-dim std: {np.median(sigma):.6f}")
