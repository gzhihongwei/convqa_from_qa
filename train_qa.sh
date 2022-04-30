#!/bin/bash

#SBATCH -J roberta-qa
#SBATCH -o output/%j.log
#SBATCH -e output/%j.err
#SBATCH -p gpu
#SBATCH -G 4
#SBATCH -t 6:00:00
#SBATCH -c 2
#SBATCH --mem 32gb

source venv/bin/activate

python3 -m torch.distributed.run --nproc_per_node 4 train_qa.py \
    # TODO: add --squad_path and or --newsqa_path
    --max_seq_len 386 \
    --stride 128 \
    --model_name_or_path deepset/roberta-base-squad2 \
    --output_dir roberta-base-squad2-expanded \
    --batch_size 96 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --epochs 2 \
    --warmup_ratio 0.2
