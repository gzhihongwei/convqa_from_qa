import argparse
import json

import benepar
import spacy

from tqdm import tqdm

from utils import largest_containing_nounphrase


def preprocess_newsqa(path: str) -> None:
    with open(path, "r") as f:
        newsqa = json.load(f)

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    for datum in tqdm(newsqa["data"]):
        context = nlp(datum["text"])
        for question in tqdm(datum["questions"], leave=False):
            if question["consensus"].get("badQuestion", False) or question[
                "consensus"
            ].get("noAnswer", False):
                continue

            try:
                initial_node = context.char_span(
                    question["consensus"]["s"],
                    question["consensus"]["e"],
                    alignment_mode="expand",
                )
                expanded_answer = str(largest_containing_nounphrase(initial_node))
                expanded_answer_start = context.text.find(expanded_answer)
                question["consensus"]["start"] = expanded_answer_start
                question["consensus"]["end"] = expanded_answer_start + len(
                    expanded_answer
                )
            except benepar.NonConstituentException:
                question["consensus"]["start"] = question["consensus"]["s"]
                question["consensus"]["end"] = question["consensus"]["e"]

    with open(path, "w") as f:
        json.dump(newsqa, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for extending noun phrases of answer spans in NewsQAv1."
    )
    parser.add_argument("squad_json", type=str, help="The NewsQA JSON to process.")
    args = parser.parse_args()
    preprocess_newsqa(args.squad_json)
