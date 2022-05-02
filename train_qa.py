import argparse
import json
import os

import pandas as pd

from datasets import Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def preprocess_qa_dataset(examples):
    inputs = tokenizer(
        examples["questions"],
        examples["contexts"],
        max_length=args.max_seq_len,
        truncation="only_second",
        stride=args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        if answer is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to cache tokenization of noun phrase expanded SQuADv2 and/or NewsQAv1."
    )
    parser.add_argument(
        "--squad_path",
        default=None,
        type=str,
        help="Path to SQuADv2 dataset with expanded noun phrase spans.",
    )
    parser.add_argument(
        "--newsqa_path",
        default=None,
        type=str,
        help="Path to NewsQAv1 dataset with expanded noun phrase spans.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=386,
        type=int,
        help="The maximum sequence length to use when tokenizing.",
    )
    parser.add_argument(
        "--stride",
        default=128,
        type=int,
        help="The stride to have if the context is too long.",
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="The checkpoint to load from.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory to where the checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size", default=96, type=int, help="The batch size per GPU to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="The number of steps to accumulate gradient updates before backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The learning rate to use while training.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="The weight decay to use on the model parameters.",
    )
    parser.add_argument(
        "--epochs", default=2, type=float, help="The number of epochs to train for."
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.2,
        type=float,
        help="The warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="The local rank of the current GPU."
    )
    args = parser.parse_args()

    assert (
        args.squad_path is not None or args.newsqa_path is not None
    ), "At least one of SQuADv2 and NewsQAv1 need to be specified"

    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", cache_dir=os.environ["TMP"]
    )
    questions = []
    contexts = []
    answers = []

    if args.squad_path:
        with open(args.squad_path, "r") as f:
            squad = json.load(f)

        for datum in squad["data"]:
            for paragraph in datum["paragraphs"]:
                for qa in paragraph["qas"]:
                    contexts.append(paragraph["context"].replace("\n", " "))
                    questions.append(qa["question"].strip())
                    answers.append(
                        None
                        if qa["is_impossible"]
                        else dict(
                            text=qa["answers"][0]["expanded"],
                            answer_start=qa["answers"][0]["expanded_start"],
                        )
                    )

    if args.newsqa_path:
        with open(args.newsqa_path, "r") as f:
            newsqa = json.load(f)

        for datum in newsqa["data"]:
            for question in datum["questions"]:
                context = datum["text"].replace("\n", " ")
                contexts.append(context)
                questions.append(question["q"].strip())
                consensus = question["consensus"]
                answers.append(
                    None
                    if any(
                        bad_key in consensus for bad_key in {"badQuestion", "noAnswer"}
                    )
                    else dict(
                        text=context[consensus["start"] : consensus["end"]],
                        answer_start=consensus["start"],
                    )
                )

    train_df = pd.DataFrame(
        dict(questions=questions, contexts=contexts, answers=answers)
    )
    raw_train_dataset = Dataset.from_pandas(train_df)
    train_dataset = raw_train_dataset.map(
        preprocess_qa_dataset,
        batched=True,
        remove_columns=raw_train_dataset.column_names,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path, cache_dir=os.environ["TMP"]
    )

    training_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        log_level="info",
        log_level_replica="warning",
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=2,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
