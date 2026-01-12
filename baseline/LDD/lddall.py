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

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    if len(tables) == 0:
        raise ValueError(f"No tables found in database {db_path}!")

    table_name = tables[0][0]
    print(f"Using table: {table_name}")

    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    embeddings = []
    for row in rows:
        emb = pickle.loads(row[3])
        embeddings.append(emb)

    conn.close()
    return np.array(embeddings, dtype=np.float64)


# =============================
# 2. LDD 计算函数（保持不变）
# =============================
def compute_ldd(
    embeddings: np.ndarray,
    subset_size: int = 512,
    num_trials: int = 5,
    reg_lambda: float = 1e-6,
    random_state: int = 42
):
    rng = np.random.default_rng(random_state)

    embeddings = np.nan_to_num(
        embeddings,
        nan=1e-10,
        posinf=1e-10,
        neginf=1e-10
    )

    embeddings = normalize(embeddings, norm='l2', axis=1)

    N = embeddings.shape[0]
    assert subset_size <= N

    ldd_values = []

    for _ in range(num_trials):
        idx = rng.choice(N, size=subset_size, replace=False)
        E = embeddings[idx]

        S = E @ E.T
        S_reg = S + reg_lambda * np.eye(subset_size)

        sign, logdet = np.linalg.slogdet(S_reg)
        if sign > 0:
            ldd_values.append(logdet)

    ldd_values = np.array(ldd_values)

    return {
        "ldd_mean": ldd_values.mean(),
        "ldd_std": ldd_values.std(),
        "ldd_all": ldd_values
    }


# =============================
# 3. 多数据集批量主流程
# =============================
if __name__ == "__main__":

    BASE_DIR = "/mnt/disk4t/heyuxuan/work/sce/output/feature"

    DATASETS = {
        "Qwen3_kmeans":    "Qwen3_kmeans_2.db",
        "Qwen3_kcenter":   "Qwen3_kcenter_2.db",
        "Qwen3_repr":      "Qwen3_repr_2.db",
        "Qwen3_random":    "Qwen3_random_2.db",
        "Llama31_kmeans":  "Llama31_kmeans_2.db",
        "Llama31_kcenter": "Llama31_kcenter_2.db",
        "Llama31_repr":    "Llama31_repr_2.db",
        "Llama31_random":  "Llama31_random_2.db",
    }

    for name, db_file in DATASETS.items():
        print("\n" + "=" * 60)
        print(f"Dataset: {name}")
        print("=" * 60)

        db_path = os.path.join(BASE_DIR, db_file)

        embeddings = read_feature(db_path)

        print(f"Total samples: {embeddings.shape[0]}")
        print(f"Embedding dim: {embeddings.shape[1]}")

        results = compute_ldd(
            embeddings,
            subset_size=512,
            num_trials=5,
            reg_lambda=1e-6
        )

        print("LDD results (Wang et al., 2024b style)")
        print(f"[{name}] Mean LDD : {results['ldd_mean']:.6f}")
        print(f"[{name}] Std  LDD : {results['ldd_std']:.6f}")
        print(f"[{name}] All  LDD : {results['ldd_all']}")
