import collections
import json
import os

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from datasets import load_dataset
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    RobertaModel,
    RobertaPreTrainedModel,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

from evaluate import eval_fn


@dataclass
class DataTrainingArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model name or path to use for initializing the model."}
    )

    context_size: str = field(
        metadata={
            "help": "The number of previous QA dialogs to include when attempting to answer question i."
        }
    )

    max_seq_len: Optional[int] = field(
        default=386,
        metadata={"help": "The maximum sequence length during tokenization.",},
    )

    stride: Optional[int] = field(default=128, metadata={"help": "The context stride."})

    quac_val_path: Optional[str] = field(
        default="", metadata={"help": "The path to the QuAC validation split JSON."}
    )


class RobertaForQUAC(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.dialog_head = nn.Linear(config.hidden_size, 6)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        followup: Optional[torch.LongTensor] = None,
        yesno: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        cls_token = sequence_output[0]

        dialog_logits = self.dialog_head(cls_token)
        followup_logits, yesno_logits = dialog_logits.split(3, dim=-1)
        followup_logits = followup_logits.squeeze(-1).contiguous()
        yesno_logits = yesno_logits.squeeze(-1).contiguous()

        total_loss = None
        if all(
            argument is not None
            for argument in {start_positions, end_positions, followup, yesno}
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(followup.size()) > 1:
                followup = followup.squeeze(-1)
            if len(yesno.size()) > 1:
                yesno = yesno.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            followup_loss = loss_fct(followup_logits, followup)
            yesno_loss = loss_fct(yesno_logits, yesno)
            total_loss = (start_loss + end_loss) / 2 + followup_loss + yesno_loss

        output = (start_logits, end_logits, followup_logits, yesno_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


def eval_quac(model, eval_dataset, tokenizer):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not training_args.no_cuda
        else torch.device("cpu")
    )
    model.to(device)
    model.eval()

    followup_map = ["y", "m", "n"]
    yesno_map = ["y", "n", "x"]

    n_best = 20
    max_answer_length = 30

    predictions = []
    for example in tqdm(eval_dataset):
        qas = example["paragraphs"][0]["qas"]
        prediction = collections.defaultdict(list)
        context = example["paragraphs"][0]["context"]

        for q_num in tqdm(range(len(qas)), leave=False):
            question = []
            start = (
                0
                if data_args.context_size is None
                else (max(0, q_num - data_args.context_size))
            )

            for i in range(start, q_num):
                question.append(qas["question"][i])
                question.append(prediction["best_span_str"][i])

            question.append(qas["question"][q_num])
            prediction["qid"].append(qas["id"][q_num])

            inputs = tokenizer(
                " ".join(question),
                context,
                max_length=data_args.max_seq_len,
                truncation="only_second",
                stride=data_args.stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt",
            )

            _ = inputs.pop("overflow_to_sample_mapping")
            offset_mapping = inputs.pop("offest_mapping").cpu().numpy().tolist()

            for i in range(len(inputs["input_ids"])):
                sequence_ids = inputs.sequence_ids(i)
                offset = offset_mapping[i]
                offset_mapping[i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                ]

            with torch.no_grad():
                outputs = model(**inputs)

            outputs = [output.cpu().numpy() for output in outputs[:4]]

            start_logits, end_logits, followup_logits, yesno_logits = outputs

            answers = []

            for i in range(len(start_logits)):
                start_logit = start_logits[i]
                end_logit = end_logits[i]
                offsets = offset_mapping[i]
                followup = followup_logits[i].argmax()
                yesno = yesno_logits[i].argmax()

                start_indexes = (-start_logit).argsort()[:n_best].tolist()
                end_indexes = (-end_logit).argsort()[:n_best].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Predicting the [CLS] token means that the model cannot answer the question
                        if start_index == end_index == 0:
                            answers.append(
                                {
                                    "text": "CANNOTANSWER",
                                    "followup": followup_map[followup],
                                    "yesno": yesno_map[yesno],
                                    "logit_score": start_logit[start_index]
                                    + end_logit[end_index],
                                }
                            )
                            continue
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answers.append(
                            {
                                "text": context[
                                    offsets[start_index][0] : offsets[end_index][1]
                                ],
                                "followup": followup_map[followup],
                                "yesno": yesno_map[yesno],
                                "logit_score": start_logit[start_index]
                                + end_logit[end_index],
                            }
                        )

            best_answer = max(answers, key=lambda x: x["logit_score"])
            prediction["best_span_str"].append(best_answer["text"])
            prediction["followup"].append(best_answer["followup"])
            prediction["yesno"].append(best_answer["yesno"])

        predictions.append(prediction)

    preds = collections.defaultdict(dict)
    for pred_idx in predictions:
        dia_id = pred_idx["qid"][0].split("_q#")[0]
        for qid, qspan, qyesno, qfollowup in zip(
            pred_idx["qid"],
            pred_idx["best_span_str"],
            pred_idx["yesno"],
            pred_idx["followup"],
        ):
            preds[dia_id][qid] = qspan, qyesno, qfollowup

    return eval_fn()


def preprocess_training_examples(examples):
    questions = []
    contexts = []
    answers = []
    answer_starts = []
    followups = []
    yesnos = []

    for qs, ans, context, followup, yesno in zip(
        examples["questions"],
        examples["orig_answers"],
        examples["context"],
        examples["followups"],
        examples["yesnos"],
    ):
        for q_num in range(len(qs)):
            question = []
            start = (
                0
                if data_args.context_size is None
                else (max(0, q_num - data_args.context_size))
            )

            for i in range(start, q_num):
                question.append(qs[i])
                question.append(ans["texts"][i])

            question.append(qs[q_num])

            questions.append(" ".join(question))
            contexts.append(context)
            answers.append(ans["texts"][q_num])
            answer_starts.append(ans["answer_starts"][q_num])
            followups.append(followup[q_num])
            yesnos.append(yesno[q_num])

    inputs = tokenizer(
        questions,
        contexts,
        max_length=data_args.max_seq_len,
        truncation="only_second",
        stride=data_args.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    followup = []
    yesno = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer_starts[sample_idx]
        end_char = start_char + len(answer)
        followup_ = followups[sample_idx]
        yesno_ = yesnos[sample_idx]
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

        followup.append(followup_)
        yesno.append(yesno_)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["followup"] = followup
    inputs["yesno"] = yesno

    return inputs


if __name__ == "__main__":
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    raw_datasets = load_dataset("quac")
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", cache_dir=os.environ["TMP"]
    )
    tokenizer.add_token("CANNOTANSWER")

    model = RobertaForQUAC.from_pretrained(data_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        with open(data_args.quac_val_path, "r") as f:
            quac_val_dataset = json.load(f)["data"]
        metrics = eval_quac(model, quac_val_dataset, tokenizer)
        trainer.log_metrics("val", metrics)
        trainer.save_metrics("val", metrics)
