import os
import sqlite3
import pickle
import numpy as np
from apricot import FacilityLocationSelection
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1. 读取 embedding
# -------------------------
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


# -------------------------
# 2. 加载数据
# -------------------------
db_path = "/mnt/disk4t/heyuxuan/work/sce/output/feature/Llama31_random_2.db"
features = read_feature(db_path)
print(f"Total samples loaded: {len(features)}")

# -------------------------
# 3. 提取 embedding + 处理 NaN / Inf
# -------------------------
embeddings = np.array(
    [features[i]['embedding'] for i in range(len(features))],
    dtype=np.float64
)

# 🔴 关键：替换 NaN / ±Inf
embeddings = np.nan_to_num(
    embeddings,
    nan=1e-10,
    posinf=1e-10,
    neginf=1e-10
)

print(f"Original embeddings shape: {embeddings.shape}")
print("Any NaN remaining:", np.isnan(embeddings).any())
print("Any Inf remaining:", np.isinf(embeddings).any())

# -------------------------
# 4. L2 归一化（用于 cosine / FL）
# -------------------------
embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
print("Embeddings normalized (L2).")

# -------------------------
# 5. Apricot Facility Location 子集选择
# -------------------------
n_samples = int(len(features) * 0.1)  # 10% 作为代表子集

selector = FacilityLocationSelection(
    n_samples=n_samples,
    metric="cosine",
    random_state=42
)

subset_embeddings = selector.fit_transform(embeddings_normalized)
print(f"Selected subset shape: {subset_embeddings.shape}")

# -------------------------
# 6. Facility Location 值计算
# -------------------------
sim_matrix = cosine_similarity(embeddings_normalized, subset_embeddings)

# 每个样本到最近代表点的相似度
max_sims = sim_matrix.max(axis=1)

fl_total = max_sims.sum()
fl_avg = max_sims.mean()

print(f"Facility Location value (total): {fl_total:.6f}")
print(f"Average FL per sample: {fl_avg:.6f}")
