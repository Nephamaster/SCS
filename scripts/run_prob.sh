#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-data/candidate/v1/candidate_doc.jsonl}"
OUTPUT_NPZ="${OUTPUT_NPZ:-output/feature/candidate_logprob_llama.npz}"
BASE_URL="${VLLM_GENERATION_BASE_URL:-http://127.0.0.1:8001/v1}"
MODEL="${VLLM_GENERATION_MODEL:-llama-3.1-8b-gen}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python -m src.extractor \
    --mode generation-probability-api \
    --input-jsonl "$INPUT_JSONL" \
    --output-npz "$OUTPUT_NPZ" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"
