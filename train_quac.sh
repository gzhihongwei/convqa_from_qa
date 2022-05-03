#!/bin/bash

#SBATCH -J roberta-quac
#SBATCH -o output/%j.log
#SBATCH -e output/%j.err
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH -t 10:00:00
#SBATCH -c 2
#SBATCH --mem 96gb
#SBATCH --nodelist gpu001

source venv/bin/activate

python3 -m torch.distributed.launch --nproc_per_node 4 run_quac.py \
    --model_name_or_path $WORK/model_path/ \
    --context_size 3 \
    --max_seq_len 386 \
    --stride 128 \
    --evaluation_strategy no \
    --output_dir quac/output_path/ \
    --do_train \
    --per_device_batch_size 32 \
    --gradient_accumulation_steps 12 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --num_train_epochs 5 \
    --warmup_ratio 0.2 \
    --log_level info \
    --log_level_replice warning \
    --logging_steps 10 \
    --save_strategy epoch \
    --fp16 \
    --dataloader_num_workers 2 \
    --optim adamw_torch \
    --report_to wandb \
    --run_name output_path

deactivate
