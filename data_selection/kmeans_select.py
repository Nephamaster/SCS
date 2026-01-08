import json
import pickle as pk
import sqlite3
import hdbscan
import os
import random
import numpy as np
from umap import UMAP
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


random.seed(46)

def read_feature(read_dir:str):
    db_path = os.path.join(read_dir)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    dataset = read_dir.split('/')[-1].replace('.db','')
    cursor.execute(f"SELECT * FROM SFT")
    rows = cursor.fetchall()
    print(len(rows))
    features = {}
    for i, row in enumerate(rows):
        document = row[1]
        start_index = row[2]
        embedding = pk.loads(row[3])
        ln_probability = row[4]
        features[i] = {
            'document': document,
            'start_index': start_index,
            'embedding': embedding,
            'ln_probability': ln_probability
        }
    conn.close()
    return features


def kmeans_cluster(data, embeddings, n_clusters:int=100, sample_size:int=10000):
    kmeans = KMeans(n_clusters=n_clusters, n_init=15)#, random_state=46
    labels = kmeans.fit_predict(embeddings)

    cluster_map = {}
    for ind, l in enumerate(labels):
        if l not in cluster_map:
            cluster_map[l] = []
        cluster_map[l].append(ind)
    
    n_select = sample_size // n_clusters
    select_inds = []
    for l in cluster_map:
        indices = cluster_map[l]
        random.shuffle(indices)
        select_inds.extend(indices[:n_select])
    
    selected_data = [data[d] for d in select_inds]
    selected_embeddings = [embeddings[d] for d in select_inds]
    if len(select_inds) < sample_size:
        all_inds = set([i for i in range(len(data))])
        select_inds = set(select_inds)
        rest_inds = list(all_inds-select_inds)
        random.shuffle(rest_inds)
        selected_data.extend([data[d] for d in rest_inds[:sample_size-len(select_inds)]])
    #     selected_embeddings.extend([embeddings[d] for d in rest_inds[:sample_size-len(select_inds)]])
    # with open('../../NovelSum/Kmeans_20.pkl', 'wb') as f:
    #     pk.dump(np.array(selected_embeddings), f)
    with open(f'../data/All/Qwen3_kmeans_2.json', 'w', encoding='utf-8') as f:
       json.dump(selected_data, f, ensure_ascii=False, indent=2)
    # with open(f'../../LLaMA-Factory/data/Qwen3_kmeans.json', 'w', encoding='utf-8') as f:
    #    json.dump(selected_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    doc_embed_file = f'../output/feature/qSFT.db'
    feature = read_feature(doc_embed_file)
    doc_embeds = []
    for i in range(len(feature)):
        doc_embeds.append(feature[i]['embedding'])

    num_nan_samples = np.isnan(doc_embeds).any(axis=1).sum()
    doc_embeds = np.nan_to_num(doc_embeds, nan=1e-10)
    print(f"包含 NaN 的样本数: {num_nan_samples}")
    Xn = normalize(doc_embeds, axis=1)
    # reducer = UMAP(n_neighbors=15, n_components=32, metric='cosine')#, random_state=46)
    # Xn_dr = reducer.fit_transform(Xn)

    with open('../data/All/SFT.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    # size_list = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000]
    # cluster_list1 = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    # cluster_list2 = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    size_list = [10000]
    cluster_list1 = [50]
    for size, cluster in zip(size_list, cluster_list1):
        print(f'---------- Sampling {size} instances by {cluster} clusters from SFT ----------')
        kmeans_cluster(data, Xn, cluster, size)
