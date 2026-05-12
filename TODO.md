# TODO

---

# QA

## 研究相关

### 你的 SCS 指标解决了什么问题？
### “现有指标会误判离群点”是什么意思？能举例说明吗？
### 为什么 HDBSCAN 适合做语义聚类？
### HDBSCAN 相比 KMeans 的优势是什么？
### 语义凝聚权重是怎么定义的？
### 为什么要融合空间距离和 LLM 生成概率？
### LLM 生成概率如何表征样本内部质量？
### 香农熵在你的方法里具体作用是什么？
### 如果一个数据集类别很多但样本质量很低，SCS 会怎么反映？
### 如果一个数据集质量很高但语义很单一，SCS 会怎么反映？
### 你说 SCS 与性能相关系数 0.985，相关性是 Pearson 还是 Spearman？
### 这个相关性是在多少组实验上算的？统计显著性如何？
### 为什么只提升 SOTA 1.3%，这个提升是否显著？
### 40w 指令数据怎么构建？如何去重、过滤毒性和低质量样本？
### DataFlow 编排了哪些过滤算子？
### 复现 4 种数据筛选方法和 12 种评估方法，分别是什么？
### 如果 embedding 模型换掉，SCS 稳定吗？
### HDBSCAN 超参如何选择？是否会影响结论？
### 你的指标是否依赖强 LLM 计算生成概率？成本如何控制？
### 如果面向中文数据，SCS 是否需要调整？


**七、算法基础和大模型基础**

### Transformer 的 self-attention 复杂度是多少？
### Multi-head attention 的作用是什么？
### RoPE 的原理是什么？
### KV Cache 的作用是什么？会占用多少显存？
### LoRA 的核心公式是什么？
### LoRA rank 怎么选？
### SFT、DPO、RLHF 的区别是什么？
### temperature、top-p、top-k 分别影响什么？
### 为什么评测时通常 temperature=0？
### beam search 和 greedy decoding 有什么区别？
### perplexity 是否能直接衡量 instruction following 能力？
### embedding 相似度常用 cosine，为什么不用欧氏距离？
### BM25 的核心思想是什么？
### cross-encoder 和 bi-encoder 的区别是什么？
### HDBSCAN 的基本思想是什么？