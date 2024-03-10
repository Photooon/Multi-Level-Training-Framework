#!/bin/bash

GPU_NUM=8
PER_DEVICE_BATCH_SIZE=64
GRADIENT_ACCUMULATION=1

# Pretrain 100K steps of GPT-2

start_time=$(date +%s)

torchrun --nproc_per_node ${GPU_NUM} --nnodes=1 --node_rank=0 train.py \
    --config_path gpt2/config.json --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --ddp_find_unused_parameters false \
    --max_steps 100000 --warmup_steps 5000 \
    --learning_rate 1e-4 --weight_decay 0.01 \
    --save_strategy steps --save_steps 10000 \
    --output_dir output/pretrained_gpt2 --overwrite_output_dir \
    --report_to tensorboard --seed 42

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Pre-Training Cost Time: $(($cost_time/60))min"

# Evaluate

python evaluate.py --model_path output/pretrained_gpt2
