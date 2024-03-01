import jsonlines
import json
import copy
import re

from tqdm import tqdm
import openai
import os
from time import sleep
import torch
import torch.nn as nn

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_open_instruct": (
        "<user>\n{instruction}\n"
        "<assistant>\n"
    ),
    "prompt_open_instruct_retrieval": (
        "<user>\nReference:{paragraph}\n{instruction}\n"
        "<assistant>\n"
    ),
    "llama_chat_prompt": (
        "[INST]{instruction}[/INST]"
    ),
    "llama_chat_prompt_retrieval": (
        "[INST]{paragraph}\n{instruction}[/INST]"
    ),
}

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]",
                          "[Retrieval]", "[Continue to Use Evidence]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]",
                        "[Utility:3]", "[Utility:4]", "[Utility:5]"]
ground_tokens_names = ["[Fully supported]",
                       "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_tokens_names:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_tokens_names:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def fix_spacing(input_text):
    # Add a space after periods that lack whitespace
    output_text = re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)
    return output_text


def postprocess(pred):
    special_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                      "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for item in special_tokens:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")

    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def preprocess_input(input_data, task):
    if task == "factscore":
        for item in input_data:
            item["instruction"] = item["input"]
            item["output"] = [item["output"]
                              ] if "output" in item else [item["topic"]]
        return input_data

    elif task == "qa":
        for item in input_data:
            if "instruction" not in item:
                item["instruction"] = item["question"]
            if "answers" not in item and "output" in item:
                item["answers"] = "output"
        return input_data

    elif task in ["asqa", "eli5"]:
        processed_input_data = []
        for instance_idx, item in enumerate(input_data["data"]):
            prompt = item["question"]
            instructions = TASK_INST[task]
            prompt = instructions + "## Input:\n\n" + prompt
            entry = copy.deepcopy(item)
            entry["instruction"] = prompt
            processed_input_data.append(entry)
        return processed_input_data


def postprocess_output(input_instance, prediction, task, intermediate_results=None):
    if task == "factscore":
        return {"input": input_instance["input"], "output": prediction, "topic": input_instance["topic"], "cat": input_instance["cat"]}

    elif task == "qa":
        input_instance["pred"] = prediction
        return input_instance

    elif task in ["asqa", "eli5"]:
        # ALCE datasets require additional postprocessing to compute citation accuracy.
        final_output = ""
        docs = []
        if "splitted_sentences" not in intermediate_results:
            input_instance["output"] = postprocess(prediction)

        else:
            for idx, (sent, doc) in enumerate(zip(intermediate_results["splitted_sentences"][0], intermediate_results["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = postprocess(sent)
                final_output += postprocessed_result[:-
                                                     1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if final_output[-1] == " ":
                final_output = final_output[:-1]
            input_instance["output"] = final_output
        input_instance["docs"] = docs
        return input_instance

def process_arc_instruction(item, instruction):
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
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output

class CosineSimilarity(nn.Module):
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


def extract_keywords(questions, task, openai_key):
    TASK_PROMPT = {
        "popqa": "Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\nquery: Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, Billy Carlson, born\n\nquestion: What is the religion of John Gwynn?\nquery: religion of John Gwynn\n\nquestion: What sport does Kiribati men's national basketball team play?\nquery: sport, Kiribati men's national basketball team play\n\nquestion: {{question}}\nquery: ",
        "pubqa": "Extract at most three keywords separated by comma from the claim as queries to extract the key information. \n\nclaim: A mother revealed to her child in a letter after her death that she had just one eye because she had donated the other to him.\nquery: a mother had one eye, donated the other eye to her child after her death.\n\nclaim: WWE wrestler Ric Flair was declared brain dead on 16 May 2019. \nquery: WWE wrestler Ric Flair, brain dead on 16 May 2019\n\nclaim: Current expenditures could likely cover the estimated costs of Medicare for All.\nquery: Current expenditures could likely cover the estimated costs of Medicare for All.\n\nclaim: Measles outbreak kills more than 1,200 in Madagascar.\nquery: Measles outbreak kills more than 1,200 in Madagascar.\n\nclaim: {{question}}\nquery:  ",
        "arc_challenge": "Extract at most three keywords separated by comma from the question as queries that can be used for searching to extract the key information. \n\nquestion: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nquery: most likely effect of increase in rotation, a planet rotates faster after a meteorite impact\n\nclaim: Jefferson's class was studying sunflowers. They learned that sunflowers are able to make their own food. Which parts of a sunflower collect most of the sunlight needed to make food? \nquery: which parts of a sunflower collect most of the sunlight needed to make food\n\nquestion: A man climbed to the top of a very high mountain. While on the mountain top, he drank all the water in his plastic water bottle and then put the cover back on. When he returned to camp in the valley, he discovered the the empty bottle had collapsed. Which of the following best explains why this happened?\nquery: best to explain the phenomenon, put the cover back of empty plastic bottle on the mountain top, empty bottle collapse in the valley\n\nquestion: There are two types of modern whales: toothed whales and baleen whales. Baleen whales filter plankton from the water using baleen, plates made of fibrous proteins that grow from the roof of their mouths. The embryos of baleen whales have teeth in their upper jaws. As the embryos develop, the teeth are replaced with baleen. Which of the following conclusions is best supported by this information?\nquery: two types of whales toothed whales and baleen whales, baleen whales filter plankton with baleen, teeth replaced with baleen\n\nquestion: {{question}}\nquery:",
    }
    assert task in TASK_PROMPT, "Your task is not included in TASK_PROMPT for a few-shot prompt template."
    openai.api_key = openai_key
    queries = []
    prompt_template = TASK_PROMPT[task]
    for question in tqdm(questions[:]):
        inputs = prompt_template.format(
            question=question
        )
        messages = [
            {"role": "user", "content": inputs},
        ]
        
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k", 
                temperature=0.1,
                messages=messages,
            )
        except openai.error.RateLimitError:
            print('Rate limit error')
            sleep(60)
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k", 
                    temperature=0.1,
                    messages=messages,
                )
            except openai.error.RateLimitError:
                print('Rate limit error')
                sleep(60)
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k", 
                    temperature=0.1,
                    messages=messages,
                )
        results = completion["choices"][0]["message"]["content"]
        queries.append(results)
    return queries

def select_relevants(strips, query, tokenizer, model, device, top_n=5):
    device = device if torch.cuda.is_available() else 'cpu'
    max_length = 512
    model = model.to(device)
    strips_data = []
    for i, p in enumerate(strips):
        if len(p.split()) < 4:
            scores = -1.0
        else:
            input_content = query + " [SEP] " + p
            inputs = tokenizer(input_content, return_tensors="pt",padding="max_length",truncation=True,max_length=max_length)
            try:
                with torch.no_grad():  
                    outputs = model(inputs["input_ids"].to(device), 
                                    attention_mask=inputs["attention_mask"].to(device))
                scores = float(outputs["logits"].cpu())
            except:
                scores = -1.0
        strips_data.append((scores, p, i))

    def take_idx(elem):
        return elem[0]
    sorted_results = sorted(strips_data, key=take_idx, reverse=True)[:]
    ctxs = [s[1] for s in sorted_results[:top_n]]
    idxs = [str(s[2]) for s in sorted_results]
    return '; '.join(ctxs), ', '.join(idxs)