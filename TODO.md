# TODO

---

# QA

## 项目动机与定义

### *你在简历里说 NovelSum、DistSum 等几何稀疏性指标容易高估离群噪声。你能具体解释一下，“真正的多样性”和“几何空间里的离群点”为什么不能等价吗？*
### *你论文里把噪声分成 intrinsic noise 和 extrinsic distributional noise。请分别举一个指令微调数据里的例子，并说明 SCS 分别靠哪个模块抑制它们。*
### *如果一个样本语法很流畅、生成概率也很高，但主题和目标指令数据分布完全不相关，SCS 会怎么处理它？为什么？*
### *SCS 最后用熵来建模多样性。为什么不用簇数量、平均簇间距离、或者簇大小方差？熵在这里表达了什么？*

## 方法细节

### *请你从输入 ShareGPT 数据开始，完整讲一遍 SCS 的计算流程，包括 embedding、clustering、generation probability、cohesion weight 和最终 entropy。*
### *生成概率在你的方法里代表什么？为什么它可以近似文本的 intrinsic quality？这个假设在哪些情况下可能失效？*
### *你代码里 `cal_gen_prob` 用的是 token log probability 的 mean。为什么不能直接用整句概率乘积？为什么 mean log-probability 更合适？*
### *你使用样本到簇中心的距离作为空间权重。为什么“靠近簇中心”就更有语义凝聚性？有没有可能簇中心反而是不好的代表？*
### *HDBSCAN 输出的噪声点标签是 `-1`。你代码里把它们映射成一个新的语义类，而不是直接丢弃。这个设计合理吗？会不会让离群点仍然贡献熵？*
### *论文公式里空间权重是 `exp(-||e_i-e_0||^2)`，但代码里看起来是 `exp(-euclidean distance)`。这是有意设计还是实现差异？两者对远离中心的样本惩罚有什么不同？*

## 聚类与表示

### *为什么选择 UMAP + HDBSCAN，而不是 KMeans、GMM 或直接在原始 embedding 空间上聚类？*
### *HDBSCAN 的 `min_cluster_size=30` 是怎么确定的？如果数据集规模从 1w 变成 100w，这个参数还应该固定为 30 吗？*
### *UMAP 降到 32 维会不会损失语义信息？你如何验证这个降维不会破坏后续聚类和 SCS 计算？*
### *你使用 mean pooling 得到样本 embedding。为什么不用 last token、CLS token、或者专门的 sentence embedding 模型？*
### *如果 embedder 和 generator 不是同一个模型，例如用 BGE 做 embedding、Llama 做生成概率，会带来什么问题？*

## 实验设计

### *你说 SCS 和下游模型性能相关性达到 0.968 / 0.985。这个相关性具体是 Pearson 还是 Spearman？为什么论文里取两者平均？*
### *实验里下游性能是 MT-Bench 和 AlpacaEval 的归一化平均。为什么这样设计？如果两个 benchmark 偏好不同，相关性还能说明问题吗？*
### *你用 GPT-5.2 做 judge。LLM-as-a-judge 本身有偏差，这会不会影响 SCS 的有效性结论？你会怎么补充验证？*
### *你们比较了 Random、KMeans、K-Center、Repr Filter 四种数据选择策略。为什么选择这四类？它们分别代表了什么数据分布特性？*
### *NovelSum 在 Llama 上相关性 0.965，和 SCS 的 0.968 很接近。你会如何证明 SCS 的提升不是实验噪声？*

## 消融与鲁棒性

### *去掉 spatial weighting 后相关性下降，去掉 probability 后也下降。你怎么解释这两个模块分别贡献了什么？*
### *如果生成概率模块使用的是 base model，而不是 fine-tuned model，会影响结果吗？论文中所谓 homologous advantage 是什么？*
### *SCS 随数据规模增长出现 plateau。这个现象从熵的角度如何解释？它对指令数据扩充有什么启发？*
### *如果一个数据集类别非常均衡，但每个类别内部都是低质量样本，SCS 会给高分还是低分？为什么？*
### *如果一个数据集只有少数几个语义簇，但每个簇内部质量极高、非常凝聚，SCS 会不会低估它对某些任务的价值？*

## 工程实现与复杂度

### *你的 SCS 计算复杂度主要瓶颈在哪里？相比 NovelSum 为什么更快？*
### *代码里特征被写入 sqlite，embedding 用 pickle 存 blob。为什么这么设计？如果数据量达到千万级，你会怎么改？*
### *当前代码逐条计算 embedding 和 generation probability。如何做 batch 化优化？需要注意哪些 padding 和 loss mask 问题？*
### *`cal_gen_prob` 里对整个拼接后的 conversation 计算 likelihood。对于 ShareGPT 多轮对话，是否应该只计算 assistant response 的概率？为什么？*
### *如果线上要做大规模数据池筛选，你会如何把 SCS 改造成可扩展的 pipeline？比如分布式 embedding、近似聚类、增量更新怎么做？*

## 针对简历表述的追问

### *你简历里写“用于预测指令微调数据集对下游模型性能的贡献”。那 SCS 是因果指标还是相关指标？如果让我基于 SCS 自动选数据，你会怎么设计闭环实验？*
### *你写“抑制内部低质噪声与外部分布离群点”。请你结合公式说明，低质噪声和分布离群点分别在哪一步被降权。*
### *你写“统一比较 12 类多样性指标作为 baseline”。这些 baseline 中哪些是 lexical-based，哪些是 distance-based，哪些是 distribution-based？各自最大缺陷是什么？*
### *如果面试官质疑：“你这个方法只是把质量和多样性混在一起，不是纯多样性指标。”你怎么回应？*
### *如果字节内部有一个面向某垂类任务的数据池，比如电商客服或广告文案，SCS 应该如何适配？是否还应该追求全局语义熵最大？*

七、算法基础和大模型基础

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