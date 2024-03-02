import argparse
import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from transformers import T5ForSequenceClassification
from transformers import T5Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler

def get_data(file):
    content = []
    label = []
    with open(file, "r", encoding="utf-8") as f:
        for i in f.readlines()[:]:
            c, l = i.split("\t")
            content.append(c)
            label.append((int(l.strip()) - 0.5) * 2)
    return content, label

def data_preprocess(file, tokenizer):
    content, label = get_data(file)
    data = pd.DataFrame({"content":content,"label":label})
    # data = shuffle(data)
    train_data = tokenizer(data.content.to_list()[:], padding = "max_length", max_length = 512, truncation=True ,return_tensors = "pt")
    train_label = data.label.to_list()[:]
    return train_data, train_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    train_file = args.train_file
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 

    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForSequenceClassification.from_pretrained("t5-large", num_labels=1)
    train_data, train_label = data_preprocess(train_file, tokenizer)

    # config
    batch_size = 12
    train = TensorDataset(train_data["input_ids"], train_data["attention_mask"], torch.tensor(train_label))
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, sampler=None)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    print(num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for i, epoch in enumerate(range(num_epochs)):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 10 == 0 and not step == 0:
                print("step: ",step, "  loss:",total_loss/(step*batch_size))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            loss = outputs.loss   
            loss.mean().backward()
            total_loss += loss.mean().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        avg_train_loss = total_loss / len(train_dataloader)      
        print("avg_loss:",avg_train_loss)
        model.save_pretrained(args.save_path + "/ep{}".format(i))
        tokenizer.save_pretrained(args.save_path + "/ep{}".format(i))

if __name__ == "__main__":
    main()

