python -m src.extractor \
    --input-jsonl data/candidate/v1/candidate_doc.jsonl \
    --output-npz output/feature/candidate_embeddings.npz \
    --base-url http://127.0.0.1:8000/v1 \
    --model llama3.1-8b-embed \
    --batch-size 256