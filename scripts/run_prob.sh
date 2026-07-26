python -m src.extractor \
    --mode generation-probability-api \
    --input-jsonl data/candidate/v1/candidate_doc.jsonl \
    --output-npz output/feature/candidate_logprob.npz \
    --base-url http://127.0.0.1:8001/v1 \
    --model llama3.1-8b-gen \
    --batch-size 256 \
    --max-len 4096