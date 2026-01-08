import os
import numpy as np
from cluster import cluster_const
from utils import *


def cal_class_prob(sem_ids:list[int],
                   ln_probs:list[float],
                   cohesion_weights:list[float]):
    """根据语义聚类结果计算凝聚概率"""
    unique_ids = sorted(list(set(sem_ids)))
    assert unique_ids == list(range(len(unique_ids))) # 确保类别编号从0严格递增
    class_gen_probs = []
    class_cohe_probs = []
    class_probs = []
    for uid in unique_ids:
        # 找到与当前编号 `uid` 相同的语义类编号的所有位置
        class_indices = [pos for pos, x in enumerate(sem_ids) if x == uid]
        class_ln_prob = [ln_probs[i] for i in class_indices]
        class_cohesion = [cohesion_weights[i] for i in class_indices]
        # 概率与凝聚度结合
        cohe_prob = np.exp(class_ln_prob) * np.array(class_cohesion)
        class_cohe_probs.append(np.sum(cohe_prob) * len(class_indices))
        # 仅生成概率
        class_gen_probs.append(np.sum(np.exp(class_ln_prob)))
        # 朴素概率
        class_probs.append(len(class_indices)*1.0/len(sem_ids))
    class_cohe_probs = np.array(class_cohe_probs)
    class_cohe_probs = class_cohe_probs / np.sum(class_cohe_probs) # 归一化
    class_gen_probs = np.array(class_gen_probs)
    class_gen_probs = class_gen_probs / np.sum(class_gen_probs) # 归一化
    class_probs = np.array(class_probs)
    return class_cohe_probs, class_gen_probs, class_probs


def cal_cohesion_weights(sem_ids:list[int], embeds):
    """根据语义聚类结果计算凝聚度"""
    unique_ids = sorted(list(set(sem_ids)))
    assert unique_ids == list(range(len(unique_ids))) # 确保类别编号从0严格递增
    class_cohesion = []
    class_centers = []
    cohesion_weights = []
    for uid in unique_ids:
        # 计算当前语义簇的中心点
        class_indices = [pos for pos, x in enumerate(sem_ids) if x == uid]
        class_embeds = [embeds[i] for i in class_indices]
        class_center = np.mean(class_embeds, axis=0)
        class_centers.append(class_center)
        # 计算语义簇中所有点到语义簇中心点的距离权重总和
        cohesion_factors = [np.exp(-cal_euclidean(class_center, embed)) for embed in class_embeds]
        cohesion_sum = np.sum(cohesion_factors)
        class_cohesion.append(cohesion_sum)
    for eid, sid in enumerate(sem_ids):
        cohesion_factor = np.exp(-cal_euclidean(class_centers[sid], embeds[eid]))
        weight = cohesion_factor / class_cohesion[sid]
        cohesion_weights.append(float(weight))
    return cohesion_weights


def get_SCS(dataset:str, data_list:list[str], features:dict, clusters:list[dict]):
    doc_lists, token_num_lists= [], []
    sem_ids = [c['sem_id'] for c in clusters]
    sem_unique_ids = set(sem_ids)
    for t in sem_unique_ids:
        doc_list, token_num_list = [], []
        for c in clusters:
            if c['sem_id'] == t:
                doc_list.append(data_list[c['doc_id']]['doc'])
                token_num_list.append(data_list[c['doc_id']]['n_tokens'])
        doc_lists.append(doc_list)
        token_num_lists.append(sum(token_num_list)/len(token_num_list))
    
    embeds = [features[c['doc_id']]['embedding'] for c in clusters]
    ln_probs = [features[c['doc_id']]['ln_probability'] for c in clusters]
    cohesion_weights = cal_cohesion_weights(sem_ids, embeds)
    class_cohe_probs, class_gen_probs, class_probs = cal_class_prob(sem_ids, ln_probs, cohesion_weights)
    sce = -np.sum(class_cohe_probs*np.log(class_cohe_probs))
    sce_wo_cohe = -np.sum(class_gen_probs*np.log(class_gen_probs))
    sce_wo_cohe_gen = -np.sum(class_probs*np.log(class_probs))

    class_gen_probs = class_gen_probs.tolist()
    class_probs = class_probs.tolist()
    class_cohe_probs = class_cohe_probs.tolist()
    class_content = {}
    docs = {}
    for i in range(len(class_probs)):
        class_content_dict = {
            'percentage':class_probs[i],
            'ln_probability':class_gen_probs[i],
            'cohesion': class_cohe_probs[i],
            'avg_token_num':token_num_lists[i],
            }
        class_content[f'class_{i}']=class_content_dict
        docs[f'class_{i}']=doc_lists[i]
    class_summarize = {
        'dataset_name': dataset,
        'SCE': sce,
        'SCE_wo_cohesion': sce_wo_cohe,
        'SCE_wo_cohesion_wo_generator': sce_wo_cohe_gen,
        'class_num': len(class_probs),
        'class_content':class_content,
    }
    docs_summarize = {
        'dataset_name': dataset,
        'docs': docs
    }
    for i in range(10):
        if os.path.exists(f"../output/result/{dataset}_SCS_{i}.json"):
            continue
        scs_path = f"../output/result/{dataset}_SCS_{i}.json"
        if os.path.exists(f"../output/result/{dataset}_doc_{i}.json"):
            continue
        doc_path = f"../output/result/{dataset}_doc_{i}.json"
        break
    save_json(class_summarize, scs_path)
    save_json(docs_summarize, doc_path)


def pipeline(dataset:str,
             generator:str='meta-llama/Llama-3.1-8B',
             embedder:str='meta-llama/Llama-3.1-8B'):
    data_file = f"../data/{dataset}.json"
    feature_file = f"../output/feature/{dataset}.db"
    cluster_file = f"../output/cluster/{dataset}.json"
    if not os.path.exists(data_file):
        data_list = data_const(dataset)
    else:
        data_list = load_json(data_file)
    if not os.path.exists(feature_file):
        features = feature_const(dataset, generator, embedder)
    else:
        features = read_feature(feature_file)
    if not os.path.exists(cluster_file):
        clusters = cluster_const(dataset)
    else:
        clusters = load_json(cluster_file)
    
    print(f'Calculate SCS of {dataset}')
    get_SCS(dataset, data_list, features, clusters)


if __name__ == '__main__':
    pipeline('demo')