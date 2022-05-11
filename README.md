# Transfer Learning Conversational QA from Expanded Extractive QA

Code repository for the COMPSCI 692F course project titled "Transfer Learning Conversational QA from Expanded Extractive QA" for Spring 2022. The technical report is available [here](https://gzhihongwei.github.io/files/wei2022transfer.pdf).

## Overview

- [Setup](#setup)
- [Expanding Answers](#expanding-answers)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citing](#citing)

## Setup

First, clone the repository.

```bash
git clone https://github.com/gzhihongwei/convqa_from_qa.git
```

Then, change directory to the root directory of the repo.

```bash
cd convqa_from_qa
```

Then run the setup script. You will need either Anaconda or Miniconda for this step.

```bash
bash setup.sh
```

This should install all dependencies and prepare the datasets for expanding answer spans, training, and evaluating.

## Expanding Answers

In order to expand the answers, run the following.

```bash
python3 preprocess/preprocess_squad.py datasets/squad/train-v2.0.json &
python3 preprocess/preprocess_newsqa.py datasets/newsqa/combined-newsqa-data-v1.json &
wait
```

The expanded answers will now be additional fields in the jsons for SQuADv2 and NewsQA.

Now, you are ready to train!

## Training

### Intermediate Fine-tuning

In order to run any intermediate fine-tuning on the expanded answers, run the below command.

```bash
python3 train_qa.py \
    --squad_path datasets/squad/train-v2.0.json \ # At least one of --squad_path and --newsqa_path have to be specified
    --newsqa_path datasets/newsqa/combined-newsqa-data-v1.json \
    --max_seq_len 386 \
    --stride 128 \
    --model_name_or_path deepset/roberta-base-squad2 \
    --output_dir <OUTPUT_DIR> \
    --batch_size 16 \
    --gradient_accumulation_steps 24 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --epochs 5 \
    --warmup_ratio 0.2
```

### Fine-tuning on QuAC

In order to run the fine-tuning on QuAC, run the below command.

```bash
python3 -m run_quac.py \
    --model_name_or_path <MODEL_NAME_OR_PATH> \ # or roberta-base or deepset/roberta-base-squad2 for baselines, or path from intermediate fine-tuning
    --context_size 3 \
    --max_seq_len 386 \
    --stride 128 \
    --evaluation_strategy no \
    --output_dir final/roberta-base \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 24 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --num_train_epochs 5 \
    --warmup_ratio 0.2 \
    --log_level info \
    --logging_steps 10 \
    --save_strategy epoch \
    --fp16 \
    --dataloader_num_workers 2 \
    --optim adamw_torch
```

## Evaluation

In order to run validation on QuAC, run the below command.

```bash
python3 run_quac.py \
    --model_name_or_path <MODEL_PATH> \
    --context_size 3 \
    --max_seq_len 386 \
    --stride 128 \
    --quac_val_path datasets/quac/val_v0.2.json \
    --output_dir <OUTPUT_DIR> \
    --do_eval
```

## Citing

If you found this code useful, please cite

```bibtex
@misc{wei2022transfer,
    url = {https://gzhihongwei.github.io/files/wei2022transfer.pdf},
    author = {Wei, George Z.},
    title = {Transfer Learning Conversational QA from Expanded Extractive QA},
    publisher = {COMPSCI 692F: Conversational AI},
    year = {2022}
}
```