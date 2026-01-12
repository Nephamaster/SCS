import os
import sqlite3
import pickle
import numpy as np
from sklearn.cluster import KMeans

# -----------------------------
# 1. 读取 embedding 的函数
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


# -----------------------------
# 2. 配置
# -----------------------------
db_path = "/mnt/disk4t/heyuxuan/work/sce/output/feature/Llama31_kmeans.db"
K = 200  # 聚类数

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
# 4. 处理 NaN / Inf（关键）
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
# 5. 计算 Cluster Inertia
# -----------------------------
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans.fit(embeddings)

inertia = kmeans.inertia_
print(f"Cluster Inertia (K={K}): {inertia:.4f}")
