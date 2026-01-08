import torch
import json
import copy
import os
import sys
import sqlite3
import pickle
import numpy as np
from tqdm import tqdm


def read_feature(read_dir:str):
    db_path = os.path.join(read_dir)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    dataset = read_dir.split('/')[-1].replace('.db','')
    cursor.execute(f"SELECT * FROM SFT")
    rows = cursor.fetchall()
    features = {}
    for i, row in enumerate(rows):
        document = row[1]
        start_index = row[2]
        embedding = pickle.loads(row[3])
        ln_probability = row[4]
        features[i] = {
            'document': document,
            'start_index': start_index,
            'embedding': embedding,
            'ln_probability': ln_probability
        }
    conn.close()
    return features


class KCenterSampling():
    def __init__(self, embeddings):
        # embeddings: numpy array of shape (N, D)
        self.embeddings = torch.from_numpy(embeddings).to(torch.float32)
        if torch.cuda.is_available():
            self.embeddings = self.embeddings.cuda()
        self.n_pool = self.embeddings.shape[0]
        self.device = self.embeddings.device

    def _compute_distances_chunked(self, query_vec, chunk_size=10000):
        """
        Compute L2 distances from all points in self.embeddings to a single query vector.
        Returns: (n_pool,) tensor of distances.
        """
        distances = []
        query_vec = query_vec.unsqueeze(0)  # (1, D)
        for i in range(0, self.n_pool, chunk_size):
            chunk = self.embeddings[i:i + chunk_size]  # (chunk_size, D)
            # L2 distance: ||a - b||_2
            dist_chunk = torch.norm(chunk - query_vec, dim=1)  # (chunk_size,)
            distances.append(dist_chunk)
            torch.cuda.empty_cache()
        return torch.cat(distances, dim=0)  # (n_pool,)

    def select(self, N):
        if N <= 0:
            return []
        if N >= self.n_pool:
            return list(range(self.n_pool))
        # Start with the farthest point from origin (or just index 0)
        # Since embeddings are already normalized (by your sample_func), norm is 1.
        # So we pick the first point arbitrarily (index 0).
        selected = [0]
        labeled_mask = torch.zeros(self.n_pool, dtype=torch.bool, device=self.device)
        labeled_mask[0] = True
        # Initialize min_dist: distance from every point to the first selected point
        min_dist = self._compute_distances_chunked(self.embeddings[0], chunk_size=10000)  # (n_pool,)
        # Greedily select N-1 more points
        for _ in tqdm(range(1, N), ncols=min(N, 100)):
            # Mask out already selected points
            min_dist[labeled_mask] = -1.0  # so they won't be selected again
            # Find the point with the largest min_dist (i.e., farthest from current set)
            next_idx = torch.argmax(min_dist).item()
            selected.append(next_idx)
            labeled_mask[next_idx] = True
            # Compute distances from all points to this new selected point
            new_dists = self._compute_distances_chunked(self.embeddings[next_idx], chunk_size=10000)
            # Update min_dist: keep the smaller distance (closer to any selected point)
            min_dist = torch.min(min_dist, new_dists)
            torch.cuda.empty_cache()
        return selected


def sample_func(embeddings, K):
    features = np.array(embeddings)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    k_center = KCenterSampling(norms)
    result = k_center.select(K)
    return result


def main(dataset:str, output_file, K):
    dataset_db = dataset.replace('-','_')
    data = json.load(fp=open(f'../data/All/SFT.json', "r"))
    feature = read_feature(f'../output/feature/{dataset_db}.db')
    text_embeddings = [feature[i]['embedding'] for i in range(len(feature))]
    res = sample_func(text_embeddings, K)
    print(res[:500])
    data_li = [data[index] for index in res]
    json.dump(obj=data_li,fp=open(output_file,"w"),indent=2,ensure_ascii=False)


if __name__ == "__main__":
    dataset = sys.argv[1]
    output_file = sys.argv[2]
    K = int(sys.argv[3])
    main(dataset, output_file, K)

# python K_Center_Greedy/kcenter_greedy_torch.py SFT ../data/All/Llama31_kcenter_2.json 10000