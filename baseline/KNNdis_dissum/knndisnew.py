import os
import sqlite3
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

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
# 配置数据库路径
# -------------------------
db_path = "/mnt/disk4t/heyuxuan/work/sce/output/feature/Llama31_kmeans.db"
features = read_feature(db_path)
print(f"Total samples loaded: {len(features)}")

# -------------------------
# 2. 提取 embedding，并处理 NaN/Inf 替换为 1e-10
# -------------------------
embeddings = np.array([features[i]['embedding'] for i in range(len(features))], dtype=np.float64)

# 替换 NaN 和 Inf 为 1e-10
embeddings = np.nan_to_num(embeddings, nan=1e-10, posinf=1e-10, neginf=1e-10)

print(f"Original embeddings shape: {embeddings.shape}")
print("Any NaN remaining:", np.isnan(embeddings).any())
print("Any Inf remaining:", np.isinf(embeddings).any())

# -------------------------
# 3. DistSumcosine
# -------------------------
embeddings_norm = normalize(embeddings, norm='l2', axis=1)
cos_sim_matrix = cosine_similarity(embeddings_norm)
dist_cos_matrix = 1 - cos_sim_matrix
dist_sum_cosine = dist_cos_matrix.sum()
print(f"DistSumcosine: {dist_sum_cosine:.4f}")

# -------------------------
# 4. DistSumL2
# -------------------------
dist_l2_matrix = euclidean_distances(embeddings)
dist_sum_l2 = dist_l2_matrix.sum()
print(f"DistSumL2: {dist_sum_l2:.4f}")

# -------------------------
# 5. KNN Distance (k=1)
# -------------------------
k = 1
nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings_norm)
distances, indices = nbrs.kneighbors(embeddings_norm)
knn_distances = distances[:, k]
avg_knn_distance = knn_distances.mean()
print(f"KNN Distance (k={k}): {avg_knn_distance:.4f}")

# -------------------------
# 6. 平均 DistSum
# -------------------------
N = embeddings.shape[0]
avg_dist_sum_cosine = dist_sum_cosine / (N * (N - 1))
avg_dist_sum_l2 = dist_sum_l2 / (N * (N - 1))
print(f"Average DistSumcosine: {avg_dist_sum_cosine:.6f}")
print(f"Average DistSumL2: {avg_dist_sum_l2:.6f}")
