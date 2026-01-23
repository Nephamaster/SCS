# SCS: Semantic Cohesion State

This is the official implementation of the paper "Beyond Geometric Sparsity: Measuring Data Diversity in LLM Instruction Tuning with Semantic Cohesion State" (Submission to ACL ARR 2026).


## Install

1. create a new conda environment
```bash
conda create -n scs python=3.11 -y
conda activate scs
```

2. install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare your data

Your instruction-tuning dataset should follow the **ShareGPT format**:

```json
[
{
    "conversations": [
    {
        "from": "user",
        "value": "..."
    },
    {
        "from": "assistant",
        "value": "..."
    },
    ...
    ]
},
...
]
```

- Save your dataset as `data/raw/<DATASET>.json`
- Refer to `data/demo.json` for a minimal working example.

### 2. Preprocess the data

From the src/ directory, run:

```bash
cd src
python datastation.py --dataset <DATASET>
```

### 3. Compute SCS

Evaluate the Semantic Cohesion State of your dataset:

```bash
python entropy.py --dataset <DATASET> \
                  --generator <GENERATOR_MODEL> \
                  --embedder <EMBEDDER_MODEL>
```

`dataset`: your IT dataset name
`generator`: Hugging Face model ID or local path for generation probability (e.g., `meta-llama/Llama-3.1-8B`)
`embedder`: Hugging Face model ID or local path for semantic embeddings (e.g., `BAAI/bge-small-en-v1.5`)

### 4. Check the result

the clusters of your dataset saved in `output/cluster/<DATASET>.json`
the intrinsic generation probabilities and semantic embeddings saved in `output/feature/<DATASET>`
the SCS score saved in `output/result/<DATASET>_SCS_0.json`

## Citation

If you find this work useful, please cite our paper:
```
@inproceedings{wu2023scs,
    title={Beyond Geometric Sparsity: Measuring Data Diversity in LLM Instruction Tuning with Semantic Cohesion State},
    author={Haiming Wu, Yuxuan He, Haiqing Zhang, Yishuo Huang, Richeng Xuan and Dawei Song},
    booktitle={Github},
    year={2026},
    url={https://github.com/Nephamaster/SCS}
}
``` 