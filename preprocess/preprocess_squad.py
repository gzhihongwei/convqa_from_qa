import argparse
import json

import benepar
import spacy

from tqdm import tqdm

from utils import largest_containing_nounphrase


def preprocess_squad(path: str) -> None:
    with open(path, "r") as f:
        squad = json.load(f)

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    for datum in tqdm(squad["data"]):
        for paragraph in tqdm(datum["paragraphs"], leave=False):
            context = nlp(paragraph["context"])
            for qa in tqdm(paragraph["qas"], leave=False):
                if "answers" in qa:
                    if not qa["is_impossible"]:
                        answers = qa["answers"]
                    else:
                        qa.pop("answers")
                if "plausible_answers" in qa and qa["is_impossible"]:
                    answers = qa["plausible_answers"]
                elif "plausible_answers" in qa and not qa["is_impossible"]:
                    print(qa)

                for answer in answers:
                    answer_start = context.text[answer["answer_start"]:].find(
                        answer["text"]) + answer["answer_start"]
                    answer_end = answer_start + len(answer["text"])
                    try:
                        initial_node = context.char_span(
                            answer_start, answer_end, alignment_mode="expand")
                        expanded_answer = str(
                            largest_containing_nounphrase(initial_node))
                        expanded_answer_start = context.text.find(
                            expanded_answer)
                        answer["expanded"] = expanded_answer
                        answer["expanded_start"] = expanded_answer_start
                    except benepar.NonConstituentException:
                        answer["expanded"] = answer["text"]
                        answer["expanded_start"] = answer["answer_start"]

    with open(path, "w") as f:
        json.dump(squad, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for extending noun phrases of answer spans in SQuADv2.")
    parser.add_argument("squad_json", type=str,
                        help="The SQuADv2 JSON to process.")
    args = parser.parse_args()
    preprocess_squad(args.squad_json)
