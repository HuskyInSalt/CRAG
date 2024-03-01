import argparse
from utils import load_file
import json

INPUT_FILES = {
    "popqa": "../retrieval_lm/eval_data/popqa_longtail_w_gs.jsonl",
    "pubqa": "../retrieval_lm/eval_data/health_claims_processed.jsonl",
    "bio": "../retrieval_lm/eval_data/factscore_unlabeled_alpaca_13b_retrieval.jsonl",
    "arc_challenge": "../retrieval_lm/eval_data/arc_challenge_processed.jsonl"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mode', type=str, default="distributed", choices=['assembled', 'distributed'],
                        help="Optional forms to build the file for self-crag.")
    parser.add_argument('--postprocess', action='store_true')
    args = parser.parse_args()

    input_file = INPUT_FILES[args.dataset]
    
    dataset = args.dataset
    input_data = load_file(input_file)
    pre = args.postprocess

    if dataset == "bio":                
        questions = []
        passages = []
        contents = []

        for item in input_data:
            question = item["question"]
            questions.append(question)
            psgs = [c["title"] + ' // ' + c["text"].strip().replace('\n', ' ') for c in item["ctxs"][:10]]
            passages.append(' [sep] '.join(psgs))
            for p in psgs:
                contents.append(question + ' [SEP] ' + p)

        with open("../data/bio/sources", 'w') as f:
            f.write('\n'.join(questions))
        with open("../data/bio/retrieved_psgs", 'w') as f:
            f.write('\n'.join(passages))
        with open("../data/bio/test_bio.txt", 'w') as f:
            f.write('\n'.join(contents))

    elif dataset == "arc_challenge":
        questions = []
        passages = []
        choice_contents = []
        contents = []
        for item in input_data:
            question = item["question"]
            questions.append(question)
            psgs = [c["title"] + ' // ' + c["text"].strip().replace('\n', ' ') for c in item["ctxs"][:10]]
            passages.append(' [sep] '.join(psgs))
            choices = [l + ": " + t for l, t in zip(item["choices"]["label"], item["choices"]["text"])]
            choice_contents.append('; '.join(choices))
            for c in item["ctxs"][:10]:
                contents.append(item["question"] + ' [SEP] ' + c["title"] + ' // ' + c["text"].strip().replace('\n', ' '))
        
        with open("../data/arc_challenge/sources", 'w') as f:
            f.write('\n'.join(questions))
        with open("../data/arc_challenge/choices", 'w') as f:
            f.write('\n'.join(choice_contents))
        with open("../data/arc_challenge/retrieved_psgs", 'w') as f:
            f.write('\n'.join(passages))
        with open("../data/arc_challenge/test_arc_challenge.txt", 'w') as f:
            f.write('\n'.join(contents))

    elif dataset == "pubqa":
        questions = []
        passages = []
        contents = []
        for item in input_data:
            question = item["question"]
            questions.append(question)
            psgs = [c["title"] + ' // ' + c["text"].strip().replace('\n', ' ') for c in item["ctxs"][:10]]
            passages.append(' [sep] '.join(psgs))
            for p in psgs:
                contents.append(question + ' [SEP] ' + p)
        
        with open("../data/pubqa/sources", 'w') as f:
            f.write('\n'.join(questions))
        with open("../data/pubqa/retrieved_psgs", 'w') as f:
            f.write('\n'.join(passages))
        with open("../data/pubqa/test_pubqa.txt", 'w') as f:
            f.write('\n'.join(contents))

    elif dataset == "popqa":
        questions = []
        passages = []
        contents = []
        for item in input_data:
            questions.append(item["question"])
            psgs = [c["text"].strip().replace('\n', ' ').replace('\t', ' ') for c in item["ctxs"][:10]]
            label = ['1' if c["title"] == item["s_wiki_title"] else '0' for c in item["ctxs"][:10]]
            passages.append(' [sep] '.join(psgs))
            contents += [item["question"] + ' [SEP] ' + p + '\t' + l for p, l in zip(psgs, label)]
        
        with open("../data/{}/sources".format(dataset), 'w') as f:
            f.write('\n'.join(questions))
        with open("../data/{}/retrieved_psgs".format(dataset), 'w') as f:
            f.write('\n'.join(passages))
        with open("../data/{}/test_popqa.txt".format(dataset), 'w') as f:
            f.write('\n'.join(contents))
    
    if pre:
        with open("../data/{}/ref/correct".format(dataset), 'r') as f:
            contexts = [l.strip()[1:] for l in f.readlines()]
        with open("../data/{}/ref/incorrect".format(dataset), 'r') as f:
            web_contexts = [l.strip()[1:] for l in f.readlines()]
        with open("../data/{}/ref/ambiguous".format(dataset), 'r') as f:
            comb_contexts = [l.strip()[:] for l in f.readlines()]
        pre_data = input_data

        for i, (item, context, web_context, comb_context) in enumerate(zip(pre_data, contexts, web_contexts, comb_contexts)):
            title = item["ctxs"][0]["title"].strip()
            if args.mode == "distributed":
                item["ctxs"] = []
                item["ctxs"] = [{"id":i, "title":title, "text":context, "score":1}]
                for web in web_context.split('; '):
                    item["ctxs"].append({"id":i, "title":title, "text":web_context, "score":0.5})
                item["ctxs"].append({"id":i, "title":title, "text":comb_context, "score":0.75})
            else:
                item["ctxs"] = [ 
                    {"id":i, "title":title, "text":context, "score":1}, 
                    {"id":i, "title":title, "text":web_context, "score":0.5}, 
                    {"id":i, "title":title, "text":comb_context, "score":0.75}
                ]
        with open("../data/{}/output/{}_selfcrag.json".format(dataset, dataset), 'w') as f:
            json.dump(pre_data, f)
        
if __name__ == "__main__":
    main()
