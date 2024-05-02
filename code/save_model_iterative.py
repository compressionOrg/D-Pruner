import os, json, itertools, bisect, gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
import sys
import torch.nn.functional as F
from sklearn.metrics import f1_score


def modify_tensor_values(tensor_list, threshold):
    for tensor in tensor_list:
        condition = tensor <= threshold
        tensor[condition] = 0


def load_model(model_name):
    print("Loading "+model_name+"...")

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_8bit=False,
        cache_dir="cache"
    ).cuda()
    model.half()

    parameters = model.named_parameters()
    with torch.no_grad():
        for name, param in parameters:
            if "self_attn.q_proj." in name:
                layer = 0 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "self_attn.k_proj." in name:
                layer = 32 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "self_attn.v_proj." in name:
                layer = 64 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "self_attn.o_proj." in name:
                layer = 96 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "mlp.gate_proj." in name:
                layer = 128 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "mlp.down_proj." in name:
                layer = 160 + int(name.split(".")[2])
                for i in range(0, 11008, 344):
                    all_values = masks[layer][:, i:i+344].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+344], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

            if "mlp.up_proj." in name:
                layer = 192 + int(name.split(".")[2])
                for i in range(0, 4096, 128):
                    all_values = masks[layer][:, i:i+128].flatten().cpu()
                    threshold = np.percentile(all_values.numpy(), 52)
                    modify_tensor_values(masks[layer][:, i:i+128], threshold)
                zero_indices = torch.nonzero(masks[layer] == 0)
                with torch.no_grad():
                    param[zero_indices[:, 0], zero_indices[:, 1]] = 0

    num0 = 0
    for p in model.parameters():
        num0 += (p == 0).sum()
    print("number of 0s:", num0)
    total_num = sum(p.numel() for p in model.parameters())
    print("total_num:", total_num)
    model.eval()
    return model


def main():
    if len(sys.argv) != 4:
        print("Usage: python save_model_iterative.py LLaMA2_HF_LOCATION IMPORTANCE_LOCATION OUTPUT_LOCATION")
        sys.exit(1)

    model_name = sys.argv[1]
    importance_location = sys.argv[2]
    save_path = sys.argv[3]

    masks = []

    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'Q/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'K/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'V/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'O/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'Gate/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'Down/' + str(i) + '.pt'), map_location='cuda:0'))
    for i in range(32):
        masks.append(torch.load(os.path.join(importance_location, 'Up/' + str(i) + '.pt'), map_location='cuda:0'))

    model = load_model(model_name)


    torch.save(model, save_path)


if __name__ == "__main__":
    main()

