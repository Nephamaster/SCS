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
- Refer to `data/raw/demo.json` for a minimal working example.

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

## SCS Data Selection

After preprocessing, feature extraction, and clustering are complete, you can select a
subset by Semantic Cohesion Weight from the repository root:

```bash
python data_selection/scs_select.py <DATASET> \
                                    --sample_size <SAMPLE_SIZE>
```

The selector sorts samples inside each semantic cluster by the Semantic Cohesion
Weight from `src/entropy.py` (`cohesion_weights`) in descending order, then selects
samples by rotating across clusters: cluster 0 top sample, cluster 1 top sample,
..., then back to cluster 0 for the next sample. The default output is saved to
`output/selection/<DATASET>_scs_<SAMPLE_SIZE>.json`.

Optional arguments:

```bash
--source raw|processed
--output <OUTPUT_JSON>
--metadata_output <METADATA_JSON>
```

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
