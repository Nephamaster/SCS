import os
import sqlite3
import pickle
import numpy as np
from sklearn.cluster import KMeans

# -----------------------------
# 1. 读取 embedding 的函数（原样保留）
# -----------------------------
def read_feature(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 查询数据库中所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)
    if len(tables) == 0:
        raise ValueError("No tables found in the database!")
    table_name = tables[0][0]  # 取第一个表
    print("Using table:", table_name)
    
    cursor.execute(f"SELECT * FROM {table_name}")
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

K = 200  # 聚类数


# =============================
# 新增：批量处理（仅新增）
# =============================
for name, db_file in DATASETS.items():

    print("=" * 70)
    print(f"Dataset: {name}")

    db_path = os.path.join(BASE_DIR, db_file)

    if not os.path.exists(db_path):
        print(f"[ERROR] DB not found: {db_path}")
        continue

    # -----------------------------
    # 3. 读取 embedding
    # -----------------------------
    features = read_feature(db_path)
    embeddings = np.array(
        [v['embedding'] for v in features.values()],
        dtype=np.float64
    )

    print(f"Total samples loaded: {embeddings.shape[0]}")
    print(f"Original embeddings shape: {embeddings.shape}")

    # -----------------------------
    # 4. 处理 NaN / Inf（原样保留）
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
    # 5. 计算 Cluster Inertia（原样保留）
    # -----------------------------
    kmeans = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10
    )

    kmeans.fit(embeddings)

    inertia = kmeans.inertia_
    print(f"Cluster Inertia (K={K}): {inertia:.4f}")
