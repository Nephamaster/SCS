# SCS 数据构建与数据选择 Pipeline

> 数据来源：`allenai/tulu-3-sft-mixture`、`m-a-p/COIG-CQIA`  
> 候选池规模：300,000 条  
> 验证集：`OpenAssistant/oasst2` 的 `validation` split  
> 数据处理框架：Data-Juicer  
> 训练框架：ms-swift  
> 目标模型：Llama-3.1-8B / Qwen3-8B-Base

## 1. 最终产物

> 注意：真实数据是放在服务器上的

```text
data/
├── raw/
│   ├── tulu-3-sft-mixture -> ../../../../data/dataset/tulu-3-sft-mixture/  # 服务器软连接
│   ├── COIG-CQIA -> ../../../../data/dataset/COIG-CQIA/  # 服务器软连接
│   └── oasst2 -> ../../../../data/dataset/oasst2/  # 服务器软连接
├── normalized/
│   ├── tulu3.jsonl
│   ├── coig_cqia.jsonl
│   └── train_merged.jsonl
├── dev/
│   └── oasst2/
│       ├── oasst2_validation.canonical.jsonl
│       └── oasst2_validation.jsonl
├── candidate/
│   └── v1/
│       ├── candidate.canonical.jsonl
│       ├── candidate_messages.jsonl
│       ├── candidate_sft.jsonl
│       ├── candidate_doc.jsonl
│       ├── candidate_metadata.jsonl
│       └── candidate_manifest.json
├── features/
│   └── candidate_v1/
│       ├── features.db
│       └── extraction_manifest.json
└── selections/
    ├── random/
    ├── kmeans/
    ├── kcenter/  # 暂不考虑
    ├── repr_filter/  # 暂不考虑
    └── scs/  # 暂不考虑
```

核心输出：

- `candidate_messages.jsonl`：300K 候选数据，包含 `sample_id`、`source`、`messages`。
- `candidate_sft.jsonl`：只包含 `messages`，直接供 ms-swift 使用。
- `candidate_doc.jsonl`：包含角色和轮次边界的 `doc`，用于 embedding、生成概率和 SCS。
- `oasst2_validation.jsonl`：OASST2 验证集，标准 `messages` JSONL。

## 2. 总体流程

```text
构建并冻结 OASST2 validation
        ↓
加载 Tulu 3 和 COIG-CQIA
        ↓
统一为 canonical JSONL
        ↓
结构检查
        ↓
删除与 OASST2 重复的数据
        ↓
训练数据内部精确去重
        ↓
训练数据内部 MinHash 近重复去重
        ↓
按来源计算 300K 配额
        ↓
Data-Juicer 在各来源内部选数
        ↓
合并并冻结 Candidate Pool v1
        ↓
提取 embedding 和生成概率
        ↓
Random / K-means
        ↓
统一物化训练子集
        ↓
ms-swift SFT + OASST2 validation
```

# 3. 原始数据格式

## 3.1 Tulu 3 原始格式

```json
{
  "id": "sample-id",
  "source": "allenai/tulu-3-sft-personas-math",
  "messages": [
    {
      "role": "user",
      "content": "Solve the following problem."
    },
    {
      "role": "assistant",
      "content": "The solution is..."
    }
  ]
}
```

处理规则：

- 保留官方 `source`。
- `sample_id` 设置为 `tulu3::<source>::<id>`。
- 删除 `source` 名称中包含 `oasst` 或 `openassistant` 的数据。
- `messages` 内容不改写，只清理首尾空白和控制字符。

## 3.2 COIG-CQIA 原始格式

```json
{
  "instruction": "请解释数据多样性。",
  "input": "结合大模型指令微调场景。",
  "output": "数据多样性是指……",
  "task_type": {
    "major": "问答"
  },
  "domain": [
    "人工智能"
  ],
  "answer_from": "human",
  "human_verified": true,
  "copyright": "..."
}
```

COIG-CQIA 整体作为一个来源：

```text
source = coig_cqia
```

转换规则：

```python
if input.strip():
    user_content = instruction.strip() + "\n\n" + input.strip()
else:
    user_content = instruction.strip()
```

转换后：

```json
{
  "sample_id": "coig_cqia::<config>::<row_id>::<hash>",
  "source": "coig_cqia",
  "messages": [
    {
      "role": "user",
      "content": "请解释数据多样性。\n\n结合大模型指令微调场景。"
    },
    {
      "role": "assistant",
      "content": "数据多样性是指……"
    }
  ],
  "metadata": {
    "task_type": {
      "major": "问答"
    },
    "domain": [
      "人工智能"
    ],
    "answer_from": "human",
    "human_verified": true,
    "copyright": "..."
  }
}
```

## 3.3 OASST2 原始格式

OASST2 的每条记录是一个消息节点，不是完整对话。

```json
{
  "message_id": "msg-001",
  "parent_id": null,
  "message_tree_id": "tree-001",
  "text": "Explain neural networks.",
  "role": "prompter",
  "rank": null,
  "lang": "en",
  "review_result": true,
  "deleted": false,
  "tree_state": "ready_for_export"
}
```

子节点：

```json
{
  "message_id": "msg-002",
  "parent_id": "msg-001",
  "message_tree_id": "tree-001",
  "text": "A neural network is...",
  "role": "assistant",
  "rank": 0,
  "lang": "en",
  "review_result": true,
  "deleted": false,
  "tree_state": "ready_for_export"
}
```

# 4. 统一 Canonical 格式

Tulu 和 COIG 转换后统一为：

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "metadata": {
    "source_dataset": "tulu3",
    "source_subset": "tulu-official-source",
    "language": "en",
    "task_type": null,
    "domain": null
  }
}
```

字段定义：

| 字段 | 用途 |
|---|---|
| `sample_id` | 全流程唯一标识 |
| `source` | 来源配额与分布统计 |
| `messages` | ms-swift SFT |
| `doc` | Candidate Pool 冻结后生成，用于 embedding、生成概率、聚类、SCS |
| `metadata` | 审计与统计 |

# 5. Candidate Pool 冻结后的 role-aware doc 格式

在 300K Candidate Pool 冻结前，只保留 `messages` 和用于去污染/去重的临时文本视图；不导出最终 `doc`。Candidate Pool 抽样完成后，才对 300K 条记录构造并导出 `doc`。

所有 `doc` 使用同一模板：

```text
[SYSTEM]
system content

<TURN_END>

[USER]
user content

<TURN_END>

[ASSISTANT]
assistant content
```

构造函数：

```python
ROLE_LABELS = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
    "tool_call": "TOOL_CALL"
}

def build_doc(messages):
    parts = []
    for message in messages:
        role = ROLE_LABELS[message["role"]]
        content = message["content"].strip()
        parts.append(f"[{role}]\n{content}")
    return "\n\n<TURN_END>\n\n".join(parts)
```

对应 JSONL：

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "doc": "[USER]\n...\n\n<TURN_END>\n\n[ASSISTANT]\n..."
}
```

# 6. OASST2 验证集构建

## 6.1 数据加载

```python
load_dataset("OpenAssistant/oasst2", split="validation")
```

## 6.2 节点过滤

仅保留：

```text
deleted == false
review_result != false
tree_state == "ready_for_export"
```

## 6.3 对话树重建

按 `message_tree_id` 分组，通过 `message_id` 和 `parent_id` 重建树。

每棵树只保留一条确定性路径：

1. 根节点必须是 `prompter`。
2. `prompter` 映射为 `user`。
3. `assistant` 保持为 `assistant`。
4. assistant 分支优先选择 `rank == 0`。
5. 无 `rank == 0` 时按 `rank` 升序选择。
6. rank 相同时按 `message_id` 排序。
7. prompter 分支按 `message_id` 排序。
8. 路径角色严格交替。
9. 路径最后一条必须是 assistant。
10. 路径以 user 结束时截断到最近一个 assistant。

## 6.4 OASST2 canonical 输出

```json
{
  "sample_id": "oasst2::<message_tree_id>::<leaf_message_id>",
  "source": "oasst2_validation",
  "messages": [
    {
      "role": "user",
      "content": "Explain neural networks."
    },
    {
      "role": "assistant",
      "content": "A neural network is..."
    }
  ],
  "metadata": {
    "message_tree_id": "tree-001",
    "leaf_message_id": "msg-002",
    "language": "en"
  }
}
```

## 6.5 ms-swift 验证集输出

`oasst2_validation.jsonl`：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain neural networks."
    },
    {
      "role": "assistant",
      "content": "A neural network is..."
    }
  ]
}
```

训练时：

```bash
--val_dataset data/dev/oasst2/oasst2_validation.jsonl
```

# 7. Data-Juicer 预处理

Data-Juicer 输入：

```text
data/normalized/train_merged.jsonl
```

单条格式：

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "dedup_text": "[USER]\n...\n\n<TURN_END>\n\n[ASSISTANT]\n...",
  "first_user_prompt": "...",
  "metadata": {
    "source_dataset": "tulu3",
    "language": "en"
  }
}
```

Data-Juicer 依次执行以下阶段。

## 7.1 结构过滤

自定义 Filter 删除：

- `messages` 为空；
- role 不属于 `system/user/assistant/tool/tool_call`；
- 第一条有效对话不是 user；
- user 和 assistant 未严格交替；
- 最后一条不是 assistant；
- content 不是字符串；
- assistant 内容为空；
- 没有非空 assistant 回复。

阶段输出格式不变。

## 7.2 与 OASST2 去重

先冻结 OASST2 validation，并建立：

```text
oasst2_doc_hashes
oasst2_prompt_hashes
oasst2_prompt_minhash_index
```

Data-Juicer 自定义 `oasst2_decontamination_filter` 删除满足任一条件的训练样本：

```text
normalized dedup_text hash 与 OASST2 相同
first_user_prompt hash 与 OASST2 相同
first_user_prompt 字符 5-gram MinHash Jaccard >= 0.80
```

OASST2 validation 永远保留，匹配到的训练样本全部删除。

删除记录：

```json
{
  "sample_id": "tulu3::source::id",
  "matched_dev_id": "oasst2::tree::leaf",
  "match_type": "prompt_minhash",
  "similarity": 0.87
}
```

## 7.3 训练数据内部精确去重

在临时 role-aware `dedup_text` 上执行：

```text
document_deduplicator
```

完整 `dedup_text` 相同，只保留一条。保留优先级：

```text
COIG human_verified == true
> COIG 其他样本
> Tulu 样本
```

同优先级时按 `source`、`sample_id` 升序保留。

## 7.4 训练数据内部近重复去重

在临时 role-aware `dedup_text` 上执行：

```text
document_minhash_deduplicator
```

固定参数：

```text
tokenization = sentencepiece
window_size = 5
num_permutations = 256
jaccard_threshold = 0.80
```

同一近重复簇保留一条，保留优先级与精确去重一致。

相同 prompt、不同 answer 不按 prompt 去重；只有完整对话达到近重复阈值时删除。

# 8. 300K 来源配额

来源集合：

```text
Tulu 官方 source 1
Tulu 官方 source 2
...
Tulu 官方 source N
coig_cqia
```

COIG-CQIA 只作为一个来源。

设去重后来源 `s` 的样本数为 `n_s`。

## 8.1 基础配额

```text
b_s = min(n_s, 500)
```

每个非空来源至少保留 500 条；不足 500 条的来源全部保留。

## 8.2 剩余配额

```text
B = Σ b_s
R = 300000 - B
r_s = max(n_s - b_s, 0)
w_s = sqrt(r_s)
```

最终配额：

```text
q_s = b_s + floor(R × w_s / Σw)
```

余数按小数部分从高到低逐条分配；来源达到可用样本上限后停止分配，并将余量分给其他来源。

验收条件：

```text
Σ q_s = 300000
0 < q_s <= n_s
每个非空来源 q_s > 0
```

去重后总样本数不足 300K 时，流程终止。

# 9. Data-Juicer 分来源选数

每个来源单独执行固定随机种子抽样：

```text
random_selector
select_num = q_s
seed = 42
```

采用随机抽样，避免 300K 候选池被单一质量模型控制。

每个来源输出：

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [...],
  "doc": "...",
  "metadata": {...}
}
```

合并后满足：

```text
总样本数 = 300000
sample_id 唯一
所有来源均有样本
```

# 10. Candidate Pool v1 输出

## 10.1 candidate.canonical.jsonl

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "doc": "[USER]\n...\n\n<TURN_END>\n\n[ASSISTANT]\n...",
  "metadata": {
    "source_dataset": "tulu3",
    "language": "en"
  }
}
```

## 10.2 candidate_messages.jsonl

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

## 10.3 candidate_sft.jsonl

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

## 10.4 candidate_doc.jsonl

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "doc": "[USER]\n...\n\n<TURN_END>\n\n[ASSISTANT]\n..."
}
```

## 10.5 candidate_metadata.jsonl

```json
{
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "row_index": 0,
  "language": "en"
}
```

所有输出按相同 `sample_id` 和相同行顺序对齐。

# 11. Candidate Manifest

`candidate_manifest.json`：

```json
{
  "candidate_version": "candidate_v1",
  "sample_count": 300000,
  "random_seed": 42,
  "doc_format_version": "role_doc_v1",
  "exact_dedup": {
    "text_key": "doc"
  },
  "minhash_dedup": {
    "text_key": "doc",
    "tokenization": "sentencepiece",
    "window_size": 5,
    "num_permutations": 256,
    "jaccard_threshold": 0.8
  },
  "oasst2_decontamination": {
    "exact_doc": true,
    "exact_prompt": true,
    "prompt_char_5gram_jaccard_threshold": 0.8
  },
  "source_distribution": {},
  "language_distribution": {},
  "removed_statistics": {},
  "file_sha256": {}
}
```

Candidate Pool 冻结后不再修改。任何内容变化都创建新版本。

# 12. 后续特征提取与数据选择

## 12.1 特征提取

输入：

```text
candidate_doc.jsonl
```

输出：

```text
features.db
```

单条特征：

```json
{
  "sample_id": "tulu3::source::id",
  "embedding": "...",
  "ln_probability": -1.27
}
```

数据库使用 `sample_id` 作为主键。

## 12.2 数据选择方法

所有方法从同一个 `candidate_v1` 中选择：

```text
Random
K-means
K-Center-Greedy
Repr Filter
SCS-guided
```

每个方法只返回：

```json
{
  "rank": 0,
  "candidate_index": 12345,
  "sample_id": "tulu3::source::id",
  "selection_score": 0.91
}
```

`candidate_index` 是冻结的 `candidate_messages.jsonl` 中的零基行号。选择算法不修改候选池；读取 embedding 和生成概率时，使用该索引读取候选池全集对应的特征行。

## 12.3 选择子集输出

选择结果只物化训练所需的 SFT JSONL 以及索引/审计信息，不再为每个选择子集生成新的 DOC：

```text
selected.sft.jsonl
selected.metadata.jsonl
selected.manifest.json
```

`selected.sft.jsonl`：

```json
{
  "candidate_index": 12345,
  "sample_id": "tulu3::source::id",
  "source": "tulu-official-source",
  "messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

# 13. 训练约束

所有 SFT 实验固定：

```text
base model
tokenizer
template
LoRA 参数
learning rate
scheduler
warmup
global batch size
epoch
max_length
loss_scale
optimizer
seed
data_seed
validation dataset
```

每个训练子集必须满足：

```text
目标样本数 = 实际编码样本数
无效 role = 0
```

训练命令：

```bash
--dataset selected.sft.jsonl
--val_dataset data/dev/oasst2/oasst2_validation.jsonl
--template llama3_2
--max_length 4096
```

# 14. 验收标准

## 14.1 OASST2 validation

| 检查项 | 标准 |
|---|---:|
| Split | `validation` |
| 每棵树路径数 | 1 |
| role 交替正确率 | 100% |
| 最后一轮为 assistant | 100% |
| 空内容 | 0 |
| 输出格式 | messages JSONL |

## 14.2 Candidate Pool

| 检查项 | 标准 |
|---|---:|
| 样本数 | 300,000 |
| `sample_id` 唯一率 | 100% |
| Tulu 来源覆盖率 | 100% |
| COIG 来源数量 | 大于 0 |
| messages 可解析率 | 100% |
| role 交替正确率 | 100% |
| 最后一轮为 assistant | 100% |
| 空 assistant | 0 |
| 内部精确重复 | 0 |
| 内部 MinHash 近重复 | 0 |
| 与 OASST2 精确重复 | 0 |
| 与 OASST2 prompt 近重复 | 0 |
| messages/doc ID 对齐 | 100% |

## 14.3 选择子集

| 检查项 | 标准 |
|---|---:|
| selected ID 数量 | 等于目标样本数 |
| 重复 ID | 0 |
| ID 不在 candidate_v1 | 0 |
| 训练阶段删样本 | 0 |
| SFT/metadata 对齐 | 100% |

---

# 15. 实施模块

```text
src/
├── data/
│   ├── tulu3_adapter.py
│   ├── coig_cqia_adapter.py
│   ├── oasst2_builder.py
│   ├── normalize.py
│   └── doc_builder.py
├── datajuicer_ops/
│   ├── messages_schema_filter.py
│   ├── oasst2_decontamination_filter.py
│   └── source_quota_selector.py
├── candidate/
│   ├── build_candidate.py
│   ├── export_candidate.py
│   └── audit_candidate.py
├── features/
│   ├── extractor.py
│   └── store.py
├── selection/
│   ├── random.py
│   ├── kmeans.py
│   ├── kcenter.py
│   ├── repr_filter.py
│   ├── scs_guided.py
│   └── materialize.py
└── metrics/
    └── scs.py
```

Data-Juicer 负责：

```text
结构过滤
OASST2 去污染
训练数据精确去重
训练数据 MinHash 去重
分来源 300K 抽样
JSONL 导出
```

自定义适配代码负责：

```text
Tulu 格式统一
COIG 格式转换
OASST2 对话树重建
role-aware doc 构造
来源配额计算
候选池和选择结果版本管理
```

# 16. 候选池构建命令

从仓库根目录执行：

```bash
python -m src.candidate.build_candidate \
  --tulu-path data/raw/tulu-3-sft-mixture \
  --coig-path data/raw/COIG-CQIA \
  --oasst-path data/raw/oasst2 \
  --minhash-tokenizer-model /share/project/wuhaiming/data/models/Llama-3.1-8B
```

三个原始数据目录按 Hugging Face 仓库相对路径精确读取：

```text
data/raw/tulu-3-sft-mixture/data/train-00000-of-00006.parquet
...
data/raw/tulu-3-sft-mixture/data/train-00005-of-00006.parquet
data/raw/COIG-CQIA/COIG-CQIA-full.jsonl
data/raw/oasst2/data/validation-00000-of-00001-1deeef95c3248fe0.parquet
```

OASST2 只读取 `validation` parquet，不读取 train 或其他 JSONL 文件。

生产运行固定使用 `sentencepiece`、`window_size=5`、`num_permutations=256`、`jaccard_threshold=0.80` 和随机种子 `42`。`--minhash-tokenizer-model` 可以指向 Llama 的 `tokenizer.model`/模型目录，也可以指向 Qwen3 的 Hugging Face tokenizer 目录；同一次 Candidate Pool 构建必须固定该路径。测试 fixture 可以显式使用 `--doc-minhash-tokenization character`，不能用于正式 Candidate Pool。

如果实验配置不执行训练数据内部的 MinHash 近重复去重，可追加：

```bash
--skip-minhash-dedup
```

该参数仍保留精确去重；启用后不需要提供 `--minhash-tokenizer-model`，manifest 会记录 `minhash_enabled=false`。

如果本次实验不执行训练集与 OASST2 validation 的去污染匹配，可追加：

```bash
--skip-oasst2-decontamination
```

该参数仍会读取并导出 OASST2 validation，manifest 会记录 `oasst2_decontamination.enabled=false`。

构建完成后执行：

```bash
python -m src.candidate.audit_candidate data/candidate/v1
```

`candidate_v1` 目录已有完整输出时命令默认复用；残缺输出会拒绝继续并要求检查。内容变化应使用新的 Candidate Pool 版本。

构建脚本会复用已经完整生成的 Candidate Pool、normalized 和 OASST2 validation 输出；检测到完整输出时跳过对应写出步骤。若需要强制重建，追加 `--overwrite`。残缺输出不会被静默复用。

# 17. vLLM 批量 embedding

候选池构建完成后，可以通过 vLLM 的 OpenAI-compatible `/v1/embeddings` 接口批量计算 `candidate_doc.jsonl` 的 embedding：

```bash
python -m src.extractor \
  --input-jsonl data/candidate/v1/candidate_doc.jsonl \
  --output-npz output/feature/candidate_embeddings.npz \
  --base-url http://127.0.0.1:8000/v1 \
  --model <served-embedding-model> \
  --batch-size 64
```

输出 `.npz` 包含两个数组：`sample_ids` 和 `embeddings`，顺序严格对齐；数组第 `i` 行对应候选池的 `candidate_index=i`。后续选择子集直接按 `candidate_index` 读取该全集特征，不重新生成 doc。该步骤只计算 embedding，不计算生成概率，也不执行 token 数量或长度审计。

# 18. vLLM 离线生成概率

生成概率可以直接通过 vLLM 的离线 Python 接口计算，不需要启动 HTTP 服务。该模式使用 `prompt_logprobs`，对输入序列第 2 个 token 到最后一个 token 的 log probability 求均值，与 `Extractor.cal_gen_prob` 的计算范围一致。

如果同时运行 embedding 服务，需要确保离线 vLLM 不与其争用同一张 GPU；生成概率模式应使用生成模型模式，不要给该实例设置 `--task embed`。

```bash
python -m src.extractor \
  --mode generation-probability \
  --input-jsonl data/candidate/v1/candidate_doc.jsonl \
  --output-npz output/feature/candidate_logprob.npz \
  --model /share/project/wuhaiming/data/models/Llama-3.1-8B \
  --batch-size 8 \
  --max-len 4096 \
  --dtype float16
```

输出 `.npz` 包含 `sample_ids` 和 `ln_probability`，数组第 `i` 行对应候选池的 `candidate_index=i`；可按 `sample_id` 与 embedding 结果合并。

# 19. vLLM API 生成概率

也可以启动生成模式的 vLLM 服务，通过 `/v1/completions` 接口计算同样的 `prompt_logprobs`。该服务不能使用 embedding 模式启动；如果 embedding 服务仍占用 8000 端口，可以使用 8001：

```bash
vllm serve \
  /share/project/wuhaiming/data/models/Llama-3.1-8B \
  --task generate \
  --served-model-name llama-3.1-8b-generate \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --max-model-len 4096
```

然后调用：

```bash
python -m src.extractor \
  --mode generation-probability-api \
  --input-jsonl data/candidate/v1/candidate_doc.jsonl \
  --output-npz output/feature/candidate_logprob.npz \
  --base-url http://127.0.0.1:8001/v1 \
  --model llama-3.1-8b-generate \
  --batch-size 8 \
  --max-len 4096
```

该接口模式要求 vLLM 版本支持 `/v1/completions` 的 `prompt_logprobs` 和 `return_token_ids` 参数；如果服务返回不支持参数，使用上一节的离线模式，或升级 vLLM。
