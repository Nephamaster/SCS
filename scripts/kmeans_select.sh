#!/usr/bin/env bash
set -euo pipefail

THREADS="${KMEANS_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export BLIS_NUM_THREADS="$THREADS"

python data_selection/kmeans_select.py \
  --input data/candidate/v1/candidate_messages.jsonl \
  --embedding-npz output/feature/candidate_embeddings_qwen.npz \
  --k-range 50 100 \
  --groups 12 \
  --sample-size 10000 \
  --batch-size 4096 \
  --max-iter 100 \
  --n-init 3 \
  --overwrite
