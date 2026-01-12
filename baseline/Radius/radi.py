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
# 2. 加载数据
# -----------------------------
db_path = "/mnt/disk4t/heyuxuan/work/sce/output/feature/Llama31_kmeans.db"
features = read_feature(db_path)

embeddings = np.array(
    [v['embedding'] for v in features.values()],
    dtype=np.float64
)

print(f"Total samples loaded: {embeddings.shape[0]}")
print(f"Original embeddings shape: {embeddings.shape}")

# -----------------------------
# 3. 处理 NaN / Inf（关键）
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
# 4. 计算 Radius（数值稳定版）
# -----------------------------
# 每个维度的标准差
sigma = np.std(embeddings, axis=0, ddof=1)

# 防止 log(0)
epsilon = 1e-12
sigma_safe = np.maximum(sigma, epsilon)

# 几何平均（论文定义）
radius = np.exp(np.mean(np.log(sigma_safe)))

print(f"Radius (geometric mean of per-dim std, stable): {radius:.6f}")

# -----------------------------
# 5. 维度级统计（辅助分析）
# -----------------------------
print(f"Mean per-dim std:   {sigma.mean():.6f}")
print(f"Std per-dim std:    {sigma.std():.6f}")
print(f"Median per-dim std: {np.median(sigma):.6f}")
