from tqdm import tqdm
import time
import torch
import torch.nn as nn

import os, json, itertools, bisect, gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
from accelerate import Accelerator
import accelerate
import torch.nn.functional as F
from sklearn.metrics import f1_score


def load_model(model_name, pruned_path):
    print("Loading "+model_name+"...")

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = torch.load(pruned_path)

    num0 = 0
    for p in model.parameters():
        num0 += (p == 0).sum()
    print("number of 0s:", num0)
    total_num = sum(p.numel() for p in model.parameters())
    print("total_num:", total_num)
    model.eval()
    return model, tokenizer


def main():
    if len(sys.argv) != 3:
        print("Usage: python save_model.py LLaMA2_HF_LOCATION PRUNED_MODEL_LOCATION")
        sys.exit(1)

    model_name = sys.argv[1]
    pruned_path = sys.argv[2]
    model, tokenizer = load_model(model_name, pruned_path)

    total_input = []
    with open('code/legal_pruning_data/MultiLegalPile_300_cleaned.txt') as file:
            for jsonObj in file:
                patientDict = json.loads(jsonObj)
                total_input.append(patientDict["text"])

    encodings = tokenizer("\n\n".join(total_input), return_tensors='pt')


    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print("perplexity:", ppl)

if __name__ == "__main__":
    main()
