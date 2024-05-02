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

    total_input, total_labels, total_candidates = [], [], []
    with open('code/legal_pruning_data/casehold_test.txt') as file:
            for jsonObj in file:
                patientDict = json.loads(jsonObj)
                total_input.append("A citing text is '" + patientDict['citing_prompt'] + "'. The correct holding statement is")
                total_labels.append(patientDict['label'])
                total_candidates.append([" " + patientDict["holding_0"], " " + patientDict["holding_1"], " " + patientDict["holding_2"], " " + patientDict["holding_3"], " " + patientDict["holding_4"]])



    total_output = []
    correct = 0
    total_num = len(total_input)
    model.eval()
    print("Start inference...")
    for i in range(total_num):
        scores = []
        batched_questions = tokenizer.batch_encode_plus([total_input[i]], add_special_tokens=False, return_tensors='pt')['input_ids']
        for j in range(5):
            batched_sentences = tokenizer.batch_encode_plus([total_input[i] + total_candidates[i][j]], add_special_tokens=False, return_tensors='pt')['input_ids']
            cuda_sentences = batched_sentences.cuda()
            batched_logprobs = F.log_softmax(model(cuda_sentences)['logits'], dim=-1).cpu()
            batched_logprobs = batched_logprobs[:, len(batched_questions[0]) - 1 : , :] / (len(batched_sentences[0]) - len(batched_questions[0]))
        
            for sentence, question, logprobs in zip(batched_sentences, batched_questions, batched_logprobs):
                answer = sentence[len(question):]
                guess = logprobs.argmax(dim=-1)
                scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).sum()))


        score1, score2, score3, score4, score5 = scores[0], scores[1], scores[2], scores[3], scores[4]
        max_score = max(score1, score2, score3, score4, score5)

        if score1 == max_score:
            if total_labels[i] == "0":
                correct += 1
            total_output.append("0")
        elif score2 == max_score:
            if total_labels[i] == "1":
                correct += 1
            total_output.append("1")
        elif score3 == max_score:
            if total_labels[i] == "2":
                correct += 1
            total_output.append("2")
        elif score4 == max_score:
            if total_labels[i] == "3":
                correct += 1
            total_output.append("3")
        elif score5 == max_score:
            if total_labels[i] == "4":
                correct += 1
            total_output.append("4")

        del cuda_sentences


    y_true = np.array([int(e) for e in total_labels])
    y_pred = np.array([int(e) for e in total_output])
    print("f1:", f1_score(y_true, y_pred, average='macro'))

if __name__ == "__main__":
    main()
