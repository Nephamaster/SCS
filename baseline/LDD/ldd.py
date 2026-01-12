import os
import sqlite3
import pickle
import numpy as np
from sklearn.preprocessing import normalize

# =============================
# 1. 读取 embedding
# =============================
def read_feature(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 自动获取数据库中的第一个表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if len(tables) == 0:
        raise ValueError(f"No tables found in database {db_path}!")
    
    table_name = tables[0][0]  # 使用第一个表
    print(f"Using table: {table_name}")

    # 读取该表数据
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    embeddings = []
    for row in rows:
        emb = pickle.loads(row[3])
        embeddings.append(emb)

    conn.close()
    return np.array(embeddings, dtype=np.float64)



# =============================
# 2. LDD 计算函数
# =============================
def compute_ldd(
    embeddings: np.ndarray,
    subset_size: int = 512,
    num_trials: int = 5,
    reg_lambda: float = 1e-6,
    random_state: int = 42
):
    """
    Wang et al., 2024b style LDD computation
    """

    rng = np.random.default_rng(random_state)

    # ---- NaN / Inf 处理 ----
    embeddings = np.nan_to_num(
        embeddings,
        nan=1e-10,
        posinf=1e-10,
        neginf=1e-10
    )

    # ---- L2 归一化（cosine similarity 前提）----
    embeddings = normalize(embeddings, norm='l2', axis=1)

    N = embeddings.shape[0]
    assert subset_size <= N, "subset_size must be <= number of samples"

    ldd_values = []

    for t in range(num_trials):
        # ---- 随机子采样（论文做法）----
        idx = rng.choice(N, size=subset_size, replace=False)
        E = embeddings[idx]  # (M, D)

        # ---- 相似矩阵（Gram matrix）----
        S = E @ E.T  # cosine similarity

        # ---- 正则化 ----
        S_reg = S + reg_lambda * np.eye(subset_size)

        # ---- log det（数值稳定）----
        sign, logdet = np.linalg.slogdet(S_reg)

        if sign <= 0:
            # 极端数值情况，直接跳过
            continue

        ldd_values.append(logdet)

    ldd_values = np.array(ldd_values)

    return {
        "ldd_mean": ldd_values.mean(),
        "ldd_std": ldd_values.std(),
        "ldd_all": ldd_values
    }


# =============================
# 3. 主流程
# =============================
if __name__ == "__main__":

    db_path = "/mnt/disk4t/heyuxuan/work/sce/output/feature/Llama31_kmeans.db"

    embeddings = read_feature(db_path)

    print(f"Total samples: {embeddings.shape[0]}")
    print(f"Embedding dim: {embeddings.shape[1]}")

    results = compute_ldd(
        embeddings,
        subset_size=512,     # 论文常用规模
        num_trials=5,        # 重复采样
        reg_lambda=1e-6
    )

    print("LDD results (Wang et al., 2024b style)")
    print(f"Mean LDD : {results['ldd_mean']:.6f}")
    print(f"Std  LDD : {results['ldd_std']:.6f}")
    print(f"All  LDD : {results['ldd_all']}")
