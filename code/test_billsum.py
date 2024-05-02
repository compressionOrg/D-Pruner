import os, json, itertools, bisect, gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time


def load_model(model_name, pruned_path):
    print("Loading "+model_name+"...")

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = torch.load(pruned_path)

    num0 = 0
    for p in model.parameters():
        num0 += (p == 0).sum()
    print("number of 0s:", num0)
    total_num = sum(p.numel() for p in model.parameters())
    print("total_num:", total_num)
    model.eval()
    generator = model.generate
    return generator, tokenizer


def main():
    if len(sys.argv) != 3:
        print("Usage: python save_model.py LLaMA2_HF_LOCATION PRUNED_MODEL_LOCATION")
        sys.exit(1)

    model_name = sys.argv[1]
    pruned_path = sys.argv[2]
    generator, tokenizer = load_model(model_name, pruned_path)

    total_input = []
    with open('code/legal_pruning_data/billsum_test.txt') as file:
            for jsonObj in file:
                patientDict = json.loads(jsonObj)
                total_input.append("A bill text is '" + patientDict["source"] + "'. The summary of the bill is '")


    print("Start inference...")
    for i in range(len(total_input)):
        fulltext = total_input[i]
        gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            generated_ids = generator(
                gen_in,
                max_new_tokens=1000,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.1, # default: 1.0
                top_k = 50, # default: 50
                top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
        text_without_prompt = generated_text[len(fulltext):]
        response_cut = text_without_prompt.split("\n")[0]
        with open('final_summaries.txt', 'a') as the_file:
            the_file.write(response_cut + '\n')

if __name__ == "__main__":
    main()
