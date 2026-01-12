import os
import sqlite3
import pickle
import numpy as np
from apricot import FacilityLocationSelection
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# 1. 数据集配置
# ==========================
BASE_DIR = "/mnt/disk4t/heyuxuan/work/sce/output/feature"
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

# ==========================
# 2. 读取 embedding
# ==========================
def read_feature(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if len(tables) == 0:
        raise ValueError(f"No tables found in database {db_path}!")
    table_name = tables[0][0]
    
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    features = {i: {'embedding': pickle.loads(row[3])} for i, row in enumerate(rows)}
    conn.close()
    return features

# ==========================
# 3. 批量处理函数
# ==========================
def process_dataset(name, file_name):
    print(f"\nProcessing dataset: {name}")
    db_path = os.path.join(BASE_DIR, file_name)
    
    # 读取 embedding
    features = read_feature(db_path)
    print(f"Total samples loaded: {len(features)}")
    
    # 提取 embedding 并处理 NaN / Inf
    embeddings = np.array([features[i]['embedding'] for i in range(len(features))], dtype=np.float64)
    embeddings = np.nan_to_num(embeddings, nan=1e-10, posinf=1e-10, neginf=1e-10)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Any NaN remaining:", np.isnan(embeddings).any())
    print("Any Inf remaining:", np.isinf(embeddings).any())
    
    # L2 归一化
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
    
    # Apricot Facility Location 子集选择
    n_samples = max(1, int(len(features) * 0.1))  # 取至少 1 个
    selector = FacilityLocationSelection(n_samples=n_samples, metric="cosine", random_state=42)
    subset_embeddings = selector.fit_transform(embeddings_normalized)
    print(f"Selected subset shape: {subset_embeddings.shape}")
    
    # Facility Location 计算
    sim_matrix = cosine_similarity(embeddings_normalized, subset_embeddings)
    max_sims = sim_matrix.max(axis=1)
    fl_total = max_sims.sum()
    fl_avg = max_sims.mean()
    
    print(f"Facility Location value (total): {fl_total:.6f}")
    print(f"Average FL per sample: {fl_avg:.6f}")
    return fl_total, fl_avg

# ==========================
# 4. 批量运行
# ==========================
results = {}
for name, file_name in DATASETS.items():
    try:
        fl_total, fl_avg = process_dataset(name, file_name)
        results[name] = {'FL_total': fl_total, 'FL_avg': fl_avg}
    except Exception as e:
        print(f"[Error] Failed to process {name}: {e}")

# ==========================
# 5. 汇总
# ==========================
print("\n===== Summary =====")
for name, res in results.items():
    print(f"{name}: FL_total={res['FL_total']:.6f}, FL_avg={res['FL_avg']:.6f}")
