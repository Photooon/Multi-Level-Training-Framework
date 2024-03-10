#!/bin/bash

GPU_NUM=8
PER_DEVICE_BATCH_SIZE=64
GRADIENT_ACCUMULATION=1

# Pretrain 10K steps for GPT-2

start_time=$(date +%s)

torchrun --nproc_per_node ${GPU_NUM} --nnodes=1 --node_rank=0 train.py \
    --config_path gpt2/config.json --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --ddp_find_unused_parameters false \
    --max_steps 5000 --warmup_steps 5000 \
    --learning_rate 1e-4 --weight_decay 0.01 \
    --save_strategy steps --save_steps 5000 \
    --output_dir output/multilevel_pretrained_gpt2/stage1 --overwrite_output_dir \
    --report_to tensorboard --seed 42

# Coalescing

cp -r small_gpt2 output/multilevel_pretrained_gpt2/small_gpt2
mv output/multilevel_pretrained_gpt2/small_gpt2 output/multilevel_pretrained_gpt2/stage2
cd ../map_tools
./scripts/Coal.sh ../example/output/multilevel_pretrained_gpt2/stage1 ../example/output/multilevel_pretrained_gpt2/stage2 ../example/output/multilevel_pretrained_gpt2/stage2
cd ../example

# Pretrain 50K steps for Small GPT-2

torchrun --nproc_per_node ${GPU_NUM} --nnodes=1 --node_rank=0 train.py \
    --model_path output/multilevel_pretrained_gpt2/stage2 --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --ddp_find_unused_parameters false \
    --max_steps 50000 --warmup_steps 0 \
    --learning_rate 4e-4 --weight_decay 0.01 \
    --save_strategy steps --save_steps 50000 \
    --output_dir output/multilevel_pretrained_gpt2/stage2 --overwrite_output_dir \
    --report_to tensorboard --seed 42

# Decoalescing and Interpolation

cd ../map_tools
./scripts/DecoIP.sh ../example/output/multilevel_pretrained_gpt2/stage2 ../example/output/pretrained_gpt2 ../example/output/multilevel_pretrained_gpt2/stage3
cd ../example

# Continue the Training of Merged GPT-2

torchrun --nproc_per_node ${GPU_NUM} --nnodes=1 --node_rank=0 train.py \
    --model_path output/multilevel_pretrained_gpt2/stage3 --bf16 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --ddp_find_unused_parameters false \
    --max_steps 50000 --warmup_steps 0 \
    --learning_rate 1e-4 --weight_decay 0.01 \
    --save_strategy steps --save_steps 10000 \
    --output_dir output/multilevel_pretrained_gpt2/stage3 --overwrite_output_dir \
    --report_to tensorboard --seed 42

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Pre-Training Cost Time: $(($cost_time/60))min"

# Evaluate

python evaluate.py --model_path output/multilevel_pretrained_gpt2/stage3