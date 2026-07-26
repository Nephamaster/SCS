MODEL=/share/project/wuhaiming/data/models/Llama-3.1-8B
TRAIN=/share/project/wuhaiming/spaces/scs/data/sft/random/random_01.jsonl
DEV=/share/project/wuhaiming/spaces/scs/data/dev/oasst2/oasst2_validation.jsonl
OUT=/share/project/wuhaiming/spaces/scs/output/models/Llama-3.1-8B-SFT-Random-01


CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4  \
swift sft \
    --output_dir "$OUT" \
    --dataset "$TRAIN" \
    --val_dataset "$DEV" \
    --model "$MODEL" \
    --check_model true \
    --model_type llama \
    --template llama3_2 \
    --loss_scale last_round \
    --disable_ignore_empty_think true \
    --add_non_thinking_prefix false \
    --truncation_strategy delete \
    --dataset_shuffle true \
    --train_dataloader_shuffle true \
    --val_dataset_shuffle false \
    --dataset_num_proc 64 \
    --load_from_cache_file false \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.001 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 4096 \
    --max_new_tokens 4096 \
    --packing false \
    --optim adamw_torch \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --weight_decay 0.0 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --eval_strategy steps \
    --save_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --save_total_limit 5 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --seed 42 \
    --data_seed 42
