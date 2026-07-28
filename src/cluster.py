import os
try:
    from .utils import read_feature, save_json
except ImportError:
    from utils import read_feature, save_json


def cluster_const(dataset:str):
    import hdbscan
    from umap import UMAP

    dataset_db = dataset.replace('-', '_')
    doc_embed_file = f'../output/feature/{dataset_db}.db'
    if os.path.exists(doc_embed_file):
        feature = read_feature(doc_embed_file)
    else:
        raise FileNotFoundError('Construct feature first !')
    doc_embeds = []
    for i in range(len(feature)):
        doc_embeds.append(feature[i]['embedding'])
    
    Xn = doc_embeds
    reducer = UMAP(n_neighbors=15, n_components=32, metric='cosine', random_state=46)
    Xn_dr = reducer.fit_transform(Xn)
    clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            metric='euclidean',
            cluster_selection_method='eom',
        )
    labels = clusterer.fit_predict(Xn_dr)
    clusters = []
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(len(doc_embeds)):
        clusters.append(
            {
                'doc_id':i,
                'sem_id':int(labels[i]) if labels[i] != -1 else n_clusters
            }
        )
    print(f'Save cluster to {dataset}.json')
    save_json(clusters, f'../output/cluster/{dataset}.json')
    
    return clusters
