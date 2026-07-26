python data_selection/kmeans_select.py \
  --input data/candidate/v1/candidate_messages.jsonl \
  --embedding-npz output/feature/candidate_embeddings_llama.npz \
  --k 50 \
  --num-groups 12 \
  --sample-size 10000 \
  --overwrite