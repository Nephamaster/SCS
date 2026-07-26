import json
import os
import csv
import re
import gc
import sqlite3
import pickle
import string
import hashlib
import numpy as np
from tqdm import tqdm
from datastation import DatasetAnalyzer


# if you want to use local models, fill it in here
# MODEL_LIST = {
#     'meta-llama/Llama-3.1-8B': '/mnt/disk4t/heyuxuan/data/models/meta-llama/Llama-3.1-8B',
#     'Qwen/Qwen3-8B-Base': '/mnt/disk4t/heyuxuan/data/models/Qwen/Qwen3-8b-Base',
#     'google-bert/bert-base-uncased': '/mnt/disk4t/heyuxuan/data/models/bert-base-uncased',
#     'FacebookAI/xlm-roberta-large': '/mnt/disk4t/heyuxuan/data/models/FacebookAI/xlm-roberta-large',
# }
MODEL_LIST = {
    'meta-llama/Llama-3.1-8B': '/share/project/wuhaiming/data/models/meta-llama/Llama-3.1-8B',
    'Qwen/Qwen3-8B-Base': '/share/project/wuhaiming/data/models/Qwen3-8B-Base',
    'google-bert/bert-base-uncased': '/share/project/wuhaiming/data/models/bert-base-uncased',
    'FacebookAI/xlm-roberta-large': '/share/project/wuhaiming/data/models/xlm-roberta-large',
}
TAGS_CHOOSE = string.ascii_uppercase


def format(text:str):
    return re.sub(r'\s+', ' ', text).strip()


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


def load_json(file:str):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file:str):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cal_linalg(a, b):
    """计算两个向量的线性距离"""
    return np.linalg.norm(a - b)


def cal_cosine(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cal_cosine_distance(a, b):
    """计算两个向量的余弦距离"""
    return 1 - cal_cosine(a, b)


def cal_euclidean(a, b):
    """计算两个向量的欧氏距离"""
    return np.sqrt(np.sum((a - b) ** 2))


def cal_manhattan(a, b):
    """计算两个向量的曼哈顿距离"""
    return np.sum(np.abs(a - b))


def cal_chebyshev(a, b):
    """计算两个向量的切比雪夫距离"""
    return np.max(np.abs(a - b))


def read_feature(read_dir:str):
    """Read feature rows written by :func:`feature_const`.

    The feature table schema is ``doc_id, embedding, ln_probability``.  Read
    the named columns explicitly instead of relying on positional indexes so
    this remains compatible with SQLite tables that store the same fields in
    a different order.
    """
    db_path = os.path.abspath(os.fspath(read_dir))
    dataset = os.path.splitext(os.path.basename(db_path))[0].replace('-', '_')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (dataset,),
        )
        table_row = cursor.fetchone()
        if table_row is None:
            raise ValueError(
                f"Feature table '{dataset}' not found in database: {db_path}"
            )

        # The table name comes from sqlite_master, but quote it before using it
        # in the SELECT statement because SQLite does not parameterize names.
        table_name = table_row[0].replace('"', '""')
        cursor.execute(
            f'SELECT doc_id, embedding, ln_probability FROM "{table_name}" '
            "ORDER BY doc_id"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    features = {}
    for i, (_doc_id, embedding_blob, ln_probability) in enumerate(rows):
        features[i] = {
            'embedding': pickle.loads(embedding_blob),
            'ln_probability': ln_probability,
        }
    return features


def feature_const(dataset:str, generator_name:str, embedder_name:str):
    import torch
    from extractor import Extractor
    data_list = load_json(f'../data/{dataset}.json')
    print('Data length: ', len(data_list))
    embedder_new_name = MODEL_LIST[embedder_name] if embedder_name in MODEL_LIST else embedder_name
    generator_new_name = MODEL_LIST[generator_name] if generator_name in MODEL_LIST else generator_name
    extractor = Extractor(generator=generator_new_name, embedder=embedder_new_name)
    features = {}
    for sid, term in tqdm(enumerate(data_list), total=len(data_list), ncols=100):
        emb = np.asarray(extractor.get_embedding(term['doc']), dtype=np.float32)
        if emb.ndim != 1 or emb.size == 0:
            raise ValueError(
                f'Feature row {sid + 1} produced an invalid embedding shape: '
                f'{emb.shape}.'
            )
        if not np.isfinite(emb).all():
            raise ValueError(
                f'Feature row {sid + 1} produced a non-finite embedding. '
                'The feature database was not written; inspect the embedder '
                'dtype or the source record before retrying.'
            )
        ln_prob = extractor.cal_gen_prob(term['doc'])
        if not np.isfinite(ln_prob):
            ln_prob = 1e-10
        features[sid] = {
            'embedding': emb,
            'ln_probability': ln_prob
        }
    del extractor.gen_model
    del extractor.emb_model
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    sql_conn = sqlite3.connect(f'../output/feature/{dataset}.db')
    dataset_db = dataset.replace('-','_')
    sql_cursor = sql_conn.cursor()
    sql_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {dataset_db} (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL,
            ln_probability REAL NOT NULL
        )
    """)
    print(f"Create table {dataset}")
    for i in tqdm(range(len(features)), total=len(features), ncols=100, desc=dataset):
        emb_blob = pickle.dumps(features[i]['embedding'])
        sql_cursor.execute(f"""
            INSERT INTO {dataset_db} (embedding, ln_probability)
            VALUES (?, ?)
        """, (emb_blob, features[i]['ln_probability']))
    sql_conn.commit()
    sql_conn.close()
    return features


def data_const(dataset:str, tokenize_model='FacebookAI/xlm-roberta-large'):
    tokenize_model = MODEL_LIST[tokenize_model] if tokenize_model in MODEL_LIST else tokenize_model
    data_analyzer = DatasetAnalyzer(dataset, tokenize_model)
    flat_list = data_analyzer.flatten()
    _, token_nums, _ = data_analyzer.tokenize()
    data_list = []
    for flat, token_num in zip(flat_list,token_nums):
        data_list.append({'doc':flat, 'n_tokens':token_num})
    save_json(data_list, f'../data/{dataset}.json')
    return data_list
