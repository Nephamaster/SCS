DATASET_LIST=(llama31_SFT_10000_kcenter llama31_SFT_10000_kmeans llama31_SFT_10000_repr SFT_10000_random)

for dataset_name in ${DATASET_LIST[@]}; do
    python novelsum.py --single_dataset_path ${dataset_name} --dense_ref_dir data/source/embedding/ --gpu_id 0 --output_csv results/${dataset_name}.csv
done