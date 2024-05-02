import copy
import os
import sys
import json
from random import sample
import random
import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.deepspeed import HfDeepSpeedConfig
import wandb


python_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)
from log import print
from arguments import ModelArguments, DataArguments, MyTrainingArguments
from mydatasets import MyDataset, get_dataset_info
from lomo_trainer import LOMOTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM


def compute_metrics(all_pred, eval_dataset, eval_prefix=None):
    golds = [ins['labels'] for ins in eval_dataset.data]
    preds = all_pred[:len(golds)]

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.float16)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model_name = model_args.model_name_or_path.split('/')[-1]
    tag_name = '_'.join([data_args.dataset_name, model_name, training_args.tag] if training_args.tag else [data_args.dataset_name, model_name])
    hparam_name = 'output'
    if training_args.optim != 'sgd':
        hparam_name += '_' + training_args.optim
    if training_args.learning_rate != 5e-4:
        hparam_name += '_lr' + str(training_args.learning_rate)
    if training_args.per_device_train_batch_size != 8:
        hparam_name += '_bs' + str(training_args.per_device_train_batch_size)
    if training_args.lr_scheduler_type != 'linear':
        hparam_name += '_' + training_args.lr_scheduler_type
    if training_args.warmup != 0:
        hparam_name += '_warmup' + str(training_args.warmup)
    if training_args.clip_grad_norm and training_args.clip_grad_norm > 0:
        hparam_name += '_clipnorm' + str(training_args.clip_grad_norm)
    if training_args.clip_grad_value and training_args.clip_grad_value > 0:
        hparam_name += '_clipgrad' + str(training_args.clip_grad_value)
    if training_args.clip_loss_value and training_args.clip_loss_value > 0:
        hparam_name += '_cliploss' + str(training_args.clip_loss_value)
    # assert training_args.clip_grad_value is None or training_args.clip_loss_value is None
    training_args.output_dir = os.path.join('outputs', tag_name, hparam_name)

    if training_args.tag == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    if training_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(training_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        wandb.init(
            name=tag_name if hparam_name == 'output' else '_'.join([tag_name, hparam_name.replace('output_', '')]),
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    ds_config = training_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    # print("path", model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = training_args.gradient_checkpointing
    # print("config", config)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        config=config,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    if data_args.domain == "medical":
        lines_1st, count = [], 0

        dataset1 = json.load(open(python_path + "/medical_pruning_data/train_set.json"))
        d_items = dataset1.items()
        for e in d_items:
            context = " ".join(e[1]["CONTEXTS"])
            if e[1]['final_decision'] == "yes":
                trailing = "the phenomenon mentioned by the question is confirmed by the abstract."
            elif e[1]['final_decision'] == "no":
                trailing = "we do not support the phenomenon mentioned by the question based on the abstract."
            elif e[1]['final_decision'] == "maybe":
                trailing = "we are uncertain whether the phenomenon mentioned by the question is supported by the abstract."
            lines_1st.append("The abstract of a biomedical research article is '" + context + "'. Here comes a question '" + e[1]['QUESTION'] + "', and please answer the question with 'yes', 'no', or 'maybe'. The answer is '" + e[1]['final_decision'] + "', which indicates that " + trailing)


        dataset2 = json.load(open(python_path + "/medical_pruning_data/ori_pqaa.json"))
        d_items = dataset2.items()
        for e in d_items:
            if count >= 150:
                break
            context = " ".join(e[1]["CONTEXTS"])
            if e[1]['final_decision'] == "yes":
                trailing = "the phenomenon mentioned by the question is confirmed by the abstract."
            elif e[1]['final_decision'] == "no":
                trailing = "we do not support the phenomenon mentioned by the question based on the abstract."
            elif e[1]['final_decision'] == "maybe":
                trailing = "we are uncertain whether the phenomenon mentioned by the question is supported by the abstract."
            lines_1st.append("The abstract of a biomedical research article is '" + context + "'. Here comes a question '" + e[1]['QUESTION'] + "', and please answer the question with 'yes', 'no', or 'maybe'. The answer is '" + e[1]['final_decision'] + "', which indicates " + trailing)
            count += 1


        lines_2nd, count = [], 0
        with open(python_path + '/medical_pruning_data/mli_train_v1.jsonl') as file:
            for jsonObj in file:
                if count >= 200:
                    break
                patientDict = json.loads(jsonObj)
                if patientDict["gold_label"] == "entailment":
                    trailing = "the hypothesis is true given the premise."
                elif patientDict["gold_label"] == "contradiction":
                    trailing = "the hypothesis is false given the premise."
                elif patientDict["gold_label"] == "neutral":
                    trailing = "the hypothesis is undetermined given the premise."
                lines_2nd.append("Premise is '" + patientDict["sentence1"] + "', and hypothesis is '" + patientDict["sentence2"] + "'. Their relationship is '" + patientDict["gold_label"] + "', and this means " + trailing)
                count += 1

        lines_3rd, count = [], 0
        with open(python_path + '/medical_pruning_data/train.txt') as file:
            for jsonObj in file:
                if count >= 200:
                    break
                patientDict = json.loads(jsonObj)
                lines_3rd.append("A question posted by a patient is '" + patientDict["question"] + "'. The summary of the question is '" + patientDict["summary"] + "'.")
                count += 1

        lines_total = lines_1st + lines_2nd + lines_3rd

    elif data_args.domain == "legal":
        lines_1st, count = [], 0
        with open(python_path + '/legal_pruning_data/train.txt') as file:
            for jsonObj in file:
                if count >= 500:
                    break
                patientDict = json.loads(jsonObj)
                if patientDict['label'] == "0":
                    idx = "first"
                elif patientDict['label'] == "1":
                    idx = "second"
                elif patientDict['label'] == "2":
                    idx = "third"
                elif patientDict['label'] == "3":
                    idx = "fourth"
                elif patientDict['label'] == "4":
                    idx = "fifth"
                else:
                    sys.exit(1)
                lines_1st.append("A citing text consisting of the context and legal citation text is '" + patientDict['citing_prompt'] + "'. Hold statement 0 is '" + patientDict["holding_0"] + "', holding statement 1 is '" + patientDict["holding_1"] + "', holding statement 2 is '" + patientDict["holding_2"] + "', holding statement 3 is '" + patientDict["holding_3"] + ", and holding statement 4 is '" + patientDict["holding_4"] + "'. Choose the correct corresponding holding statement. The correct answer is holding statement " + patientDict['label'] + ", which is the " + idx + " statement.")
                count += 1

        lines_2nd, count = [], 0
        with open(python_path + '/legal_pruning_data/billsum_train.txt') as file:
            for jsonObj in file:
                if count >= 500:
                    break
                patientDict = json.loads(jsonObj)
                lines_2nd.append("A bill text is '" + patientDict["source"] + "'. The summary of the bill is '" + patientDict["summary"] + "'.")
                count += 1

        lines_total = lines_1st + lines_2nd

    train_lines, eval_lines = [], []
    for i in range(len(lines_total)):
        train_lines.append(lines_total[i])
    
    for i in range(10):
        eval_lines.append(lines_total[i])

    train_dataset = MyDataset(data_args, tokenizer, train_lines, split="train")
    eval_dataset = MyDataset(data_args, tokenizer, eval_lines, split="validation")

    # ========== 4. Initialize our Trainer. ==========
    trainer = LOMOTrainer(
        model=model,
        training_args=training_args,
        data_collator={'train': DataCollatorForCauselLM(tokenizer, max_length=data_args.data_max_length, padding_side='left'),
                       'eval': EvalDataCollatorForCauselLM(tokenizer, max_length=data_args.data_max_length, padding_side='left')},
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if training_args.do_train:
        trainer.train()
    else:
        trainer.eval(trainer.global_step, 0, trainer.eval_dataset, trainer.eval_dataloader, 'zero-shot')


if __name__ == "__main__":
    train()
