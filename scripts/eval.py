#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jsonlines
import numpy as np
from tqdm import tqdm
import json
import argparse
from tqdm import tqdm
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy

def preprocess_input_data(dataset, task=None):
    new_data = []
    if task in TASK_INST:
        instruction = TASK_INST[task]
    else:
        instruction = None
    for item in dataset:
        if task == "arc_c":
            choices = item["choices"]
            answer_labels = {}
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        else:
            prompt = instruction + "\n\n## Input:\n\n" + \
                item["question"] if instruction is not None else item["question"]
            item["instruction"] = prompt
        new_data.append(item)

    return new_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default=None)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--task', type=str)
    # Decoding hyperparams
    parser.add_argument('--metric', type=str, help="metric to be used during evaluation")
    args = parser.parse_args()

    input_path = args.input_file
    if input_path.endswith(".json"):
        input_data = json.load(open(input_path))
    else:
        input_data = load_jsonlines(input_path)

    input_data = preprocess_input_data(
        input_data, task=args.task)
    eval_file = args.eval_file
    with open(eval_file, 'r') as f:
        resps = [l.strip()[:] for l in f.readlines()]    
    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    count = 0
    for i, (pred, row) in tqdm(enumerate(zip(resps[:], input_data[:]))):
        pred = pred.strip()
        prompts.append(None)
        preds.append(pred)
        all_results.append(None)
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if args.metric == "accuracy":
            metric_result = accuracy(pred, row["output"])

        elif args.metric == "match":
            if "SUPPORTS" in pred:
                pred = "true"
            elif "REFUTES" in pred:
                pred = "false"
            metric_result = match(pred, row["answers"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        if count % 10 == 0:
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}
        count += 1

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  args.metric, "metric_mean": np.mean(metric_results), "scores": scores}

    print("Final result: {0}".format(np.mean(metric_results)))

if __name__ == "__main__":
    main()