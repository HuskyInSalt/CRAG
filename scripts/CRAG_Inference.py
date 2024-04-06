
import argparse
import logging

import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler

from vllm import LLM, SamplingParams
from transformers import T5Tokenizer, T5ForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM

# logger = logging.getLogger(__name__)

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

'''
task = ### Instruction: + TASK_INST[task] + ## Input: + question + ### Response:
task = ### Instruction: + TASK_INST[task] + ## Input: + question + choices + ### Response:
'''
def format_prompt(i, task, question, paragraph=None):
    # if task == "bio":
    #     prompt = "### Instruction:\n{0}\n\n### Response:\n".format(question)
    #     if paragraph is not None:
    #         prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    # else:
    instruction = TASK_INST[task] if task in TASK_INST else None
    instruction = instruction + "\n\n## Input:\n\n" + question if instruction is not None else question
    if task == "arc_challenge":
        with open("../data/arc_challenge/choices", 'r') as f:
            choices = f.readlines()[i].strip()
        choices = choices.replace("A: ", "\nA: ")
        choices = choices.replace("B: ", "\nB: ")
        choices = choices.replace("C: ", "\nC: ")
        choices = choices.replace("D: ", "\nD: ")
        choices = choices.replace("E: ", "\nE: ")
        instruction += choices

    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    if paragraph is not None:
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt

def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

def data_preprocess(file):
    # with_label = True
    with_label = False
    queries = []
    passages = []
    tmp_psgs = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for line in f.readlines()[:]:
                c, l = line.strip().split("\t")
                q, p = c.split(' [SEP] ')
                if queries == []:
                    queries.append(q)
                    tmp_psgs = [p]
                else:
                    if q != queries[-1]:
                        passages.append(' [sep] '.join(tmp_psgs))
                        queries.append(q)
                        tmp_psgs = [p]
                    else:
                        tmp_psgs.append(p)
            passages.append(' [sep] '.join(tmp_psgs))
        else:
            for line in f.readlines():
                c = line.strip()
                if c.endswith('[SEP]'):
                    c += ' '
                q, p = c.split(' [SEP] ')
                if queries == []:
                    queries.append(q)
                    tmp_psgs = [p]
                else:
                    if q != queries[-1]:
                        passages.append(' [sep] '.join(tmp_psgs))
                        queries.append(q)
                        tmp_psgs = [p]
                    else:
                        tmp_psgs.append(p)
            passages.append(' [sep] '.join(tmp_psgs))
    return queries, passages

def get_evaluator_data(file):
    with_label = False
    # with_label = True
    content = []
    label = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for line in f.readlines()[:]:
                c, l = line.split("\t")
                content.append(c)
                label.append((int(l.strip()) - 0.5) * 2)
            return content, label
        else:
            for line in f.readlines():
                content.append(line.strip())
            return content, None

def inference(tokenizer, model, file, device=torch.device("cpu"), n_docs=10):
    model.eval()
    content, label = get_evaluator_data(file)

    preds = []
    scores = []

    for c in tqdm(content[:]):
        if c.strip().endswith('[SEP]'):
            preds.append(-1)
            scores.append(-1.0)
            continue
        test = tokenizer(c, return_tensors="pt",padding="max_length",max_length=512)
        with torch.no_grad():  
            outputs = model(test["input_ids"].to(device), 
                            attention_mask=test["attention_mask"].to(device))
        pred_flat = 1 if outputs["logits"].cpu() > 0 else -1
        scores.append(float(outputs["logits"].cpu()))
        preds.append(pred_flat)
    return scores

def process_flag(scores, n_docs, threshold1, threshold2):
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append('2')
        elif score >= threshold2:
            flags.append('1')
        else:
            flags.append('0')

    tmp_flag = []
    identification_flag = []
    for i, f in enumerate(flags):
        tmp_flag.append(f)
        if i % n_docs == n_docs - 1:
            if '2' in tmp_flag:
                identification_flag.append(2)
            elif '1' in tmp_flag:
                identification_flag.append(1)
            else:
                identification_flag.append(0)
            tmp_flag = []
    return identification_flag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str)
    parser.add_argument('--evaluator_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--internal_knowledge_path', type=str)
    parser.add_argument('--external_knowledge_path', type=str)
    parser.add_argument('--combined_knowledge_path', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--method', type=str, default="default", choices=['rag', 'crag', 'no_retrieval'])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=-1,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--upper_threshold", type=float, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--lower_threshold", type=float, default=10,
                        help="Number of documents to retrieve per questions")
    args = parser.parse_args()
    args.lower_threshold = -args.lower_threshold

    # assert 1==0
    tokenizer = T5Tokenizer.from_pretrained(args.evaluator_path)
    model = T5ForSequenceClassification.from_pretrained(args.evaluator_path, num_labels=1)

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # assert 1==0
    queries, passages = data_preprocess(args.input_file)

    # assert 1==0
    scores = inference(
        tokenizer=tokenizer, 
        model=model, 
        file=args.input_file,
        device=device, 
        n_docs=args.ndocs
    )

    # str_scores = [str(s) for s in scores]
    # with open('../data/pubqa/score_log', 'w') as f:
    #     f.write('\n'.join(str_scores))

    # with open('../data/arc_challenge/score_log', 'r') as f:
    #     scores = [float(l.strip()) for l in f.readlines()]
    identification_flag = process_flag(scores, args.ndocs, args.upper_threshold, args.lower_threshold)

    if args.method == 'rag':
        paragraphs = passages
    elif args.method == 'crag':
        with open(args.internal_knowledge_path, 'r') as in_f, open(args.external_knowledge_path, 'r') as ex_f, open(args.combined_knowledge_path, 'r') as comb_f:
            internal_paragraphs = [l.strip() for l in in_f.readlines()]
            external_paragraphs = [l.strip() for l in ex_f.readlines()]
            combined_paragraphs = [l.strip() for l in comb_f.readlines()]

        paragraphs = []
        for flag, i, e, c in zip(identification_flag, internal_paragraphs, external_paragraphs, combined_paragraphs):
            if flag == 0:
                paragraphs.append(e) # incorrect
            elif flag == 1:
                paragraphs.append(c) # ambiguous
            elif flag == 2:
                paragraphs.append(i) # correct
    
    preds = []
    generator = LLM(model=args.generator_path, dtype="half")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
    if args.method != 'no_retrieval':
        for i, (q, p) in tqdm(enumerate(zip(queries, paragraphs))):
            prompt = format_prompt(i, args.task, q, p)
            pred = generator.generate([prompt], sampling_params)
            preds.append(postprocess_answer_option_conditioned(pred[0].outputs[0].text))
    else:
        for i, q in tqdm(enumerate(queries)):
            p = None
            prompt = format_prompt(i, args.task, q, p)
            pred = generator.generate([prompt], sampling_params)
            preds.append(postprocess_answer_option_conditioned(pred[0].outputs[0].text))

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(preds))


if __name__ == '__main__':
    main()
