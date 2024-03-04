import argparse
from tqdm import tqdm

from bs4 import BeautifulSoup

import torch
import torch.nn as nn
from transformers import T5ForSequenceClassification, T5Tokenizer
from utils import select_relevants

def extract_strips_from_psg(psg, mode="excerption"):
    if mode == 'fixed_num':
        final_strips = []
        window_length = 50
        words = psg.split(' ')
        buf = []
        for w in words:
            buf.append(w)
            if len(buf) == window_length:
                final_strips.append(' '.join(buf))
                buf = []
        if buf != []:
            if len(buf) < 10:
                final_strips[-1] += (' ' + ' '.join(buf))
            else:
                final_strips.append(' '.join(buf))
        return final_strips
    if mode == 'excerption':
        num_concatenate_strips = 3
        question_strips = psg.split('?')
        origin_strips = []
        for qs in question_strips:
            origin_strips += qs.split('. ')
        strips = []
        for s in origin_strips:
            if s in strips:
                continue
            if strips == []:
                strips.append(s)
            else:
                if len(s.split()) > 5:
                    strips.append(s)
                else:
                    strips[-1] += s
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(' '.join(buf))
                buf = []
        if buf != []:
            final_strips.append(' '.join(buf))
        return final_strips
    elif mode == 'selection':
        return [psg]

def knowledge_refinement(psgs, queries, output_path, model_name, device, decompose_mode):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
    top_n = 3 if decompose_mode == "selection" else 6

    output_results = []
    output_idxs = []
    progress_bar = tqdm(range(len(queries[:])))
    for psg, query in zip(psgs[:], queries[:]):
        results = ''
        strips = []
        for p in psg:
            strips += extract_strips_from_psg(psg=p, mode=decompose_mode)

        results, idxs = select_relevants(
                strips=strips, 
                query=query,
                tokenizer=tokenizer,
                model=model,
                device=device,
                top_n=top_n,
            )
        
        results = results.replace('\n', ' ')
        output_results.append(results)
        output_idxs.append(idxs)
        progress_bar.update(1)

    with open(output_path, 'w') as f:
        f.write('#')
        f.write('\n#'.join(output_results))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--input_queries', type=str)
    parser.add_argument('--input_retrieval', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--decompose_mode', type=str, default="selection", choices=['selection', 'excerption', 'fixed_num'],
                        help="Optional strategies to decompose the retrieval results")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    with open(args.input_retrieval, 'r') as psg_f, open(args.input_queries, 'r') as query_f:
        passages = [p.strip().split('[sep]') for p in psg_f.readlines()]
        queries = [q.strip() for q in query_f.readlines()]

    results = knowledge_refinement(passages, queries, args.output_file, args.model_path, args.device, args.decompose_mode)


if __name__ == '__main__':
    main()
