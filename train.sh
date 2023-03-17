# export CUDA_VISIBLE_DEVICES=3,5,6,7

torchrun --nproc_per_node=8 --master_port=10777 train.py \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir /srv/local/data/zhenlin4/stanford_alpaca \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True