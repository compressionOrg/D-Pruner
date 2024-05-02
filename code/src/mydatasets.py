import os
import copy
import json
import random
from tqdm import tqdm
from typing import Callable, Any

from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from log import print
from prompts import QuestionPart, Exemplar, idx_to_ltr

IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


class MyDataset(Dataset):
    def __init__(self, data_args, tokenizer, lines, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        random.shuffle(lines)
        self.lines = lines
        self.split = split
        self.sample_size = 500000
        # self.prompt_type = dataset_info.prompt_type

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            self.data = self.process(lines, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)


    def process(self, lines, save_file):
        data = []
        for instance in tqdm(lines):
            source = instance

            def _tokenize_fn(source, target):
                source_tokenized = self.tokenizer.encode(source)

                labels = copy.deepcopy(source_tokenized)
                return source_tokenized, labels

            if self.split == 'train':
                input_ids, labels = _tokenize_fn(source, source)
            else:
                input_ids, labels = _tokenize_fn(source, source)

            data.append({'input_ids': input_ids,
                         'labels': labels})

        if self.sample_size > 0 and len(data) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            possible_idxs = list(range(len(data)))
            sampled_idxs = random.sample(possible_idxs, self.sample_size)
            data = [data[i] for i in sampled_idxs]
            print(f'Sampled {self.sample_size} examples from {len(possible_idxs)} examples.')

        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }


@dataclass
class DatasetInfo:
    path: str = None
    exemplar_split: str = None
    eval_split: str = None
    test_split: str = None
    extractor: Callable = Any
    name: str = None
    data_dir: str = None
    sample_size: int = -1
    prompt_type: str = 'brown'


def get_dataset_info(dataset_name):
    if dataset_name == 'boolq':
        return DatasetInfo(
            path="super_glue",
            name="boolq",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['passage']} {row['question']}",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )

    elif dataset_name == 'multirc':
        return DatasetInfo(
            path="super_glue",
            name="multirc",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['paragraph']}",
                    ),
                    QuestionPart(
                        f"{row['question']}",
                        tag='Question'
                    ),
                    QuestionPart(
                        f'I found this answer "{row["answer"]}". Is that correct? Yes or No?',
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'rte':
        return DatasetInfo(
            path="super_glue",
            name="rte",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['premise']}\nDoes this mean that \"{row['hypothesis']}\" is true? Yes or No?",
                    ),
                ],
                "choices": [
                    'Yes', 'No'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'wic':
        return DatasetInfo(
            path="super_glue",
            name="wic",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"Does the word \"{row['word']}\" have the same meaning in these two sentences? Yes, No?\n{row['sentence1']}\n{row['sentence2']}",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'wsc':
        return DatasetInfo(
            path="super_glue",
            name="wsc",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['text']}\nIn the previous sentence, does the pronuon \"{row['span2_text']}\" refer to \"{row['span1_text']}\"? Yes or No?",
                    ),
                ],
                "choices": [
                    'No', 'Yes'
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'copa':
        return DatasetInfo(
            path="super_glue",
            name="copa",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            prompt_type='natural',
            extractor=lambda row: {
                "parts": [
                    QuestionPart(
                        f"{row['premise']} so " if row['question'] == 'effect' else f"{row['premise']} because ",
                    ),
                ],
                "choices": [
                    row['choice1'], row['choice2']
                ],
                "answer_idx": int(row["label"])
            }
        )
    elif dataset_name == 'record':
        return DatasetInfo(
            path="super_glue",
            name="record",
            exemplar_split="train",
            eval_split="validation",
            sample_size=1000,
            extractor=process_record
        )
    else:
        raise NotImplementedError


def process_record(row):
    def record_clean_choices(row):
        if len(row['answers']) == 1:
            return row['entities'], row['entities'].index(row['answers'][0])

        new_entities = []
        for entity in row['entities']:
            if entity in row['answers'][1:]:
                continue
            new_entities.append(entity)
        return new_entities, new_entities.index(row['answers'][0])

    choices, answer_idx = record_clean_choices(row)
    return {
                "parts": [
                    QuestionPart(
                        "{}\n{}\nQuestion: What is the \"@placeholder\"?".format(row['passage'].replace('@highlight\n', '- '), row['query']),
                    ),
                ],
                "choices": choices,
                "answer_idx": answer_idx
            }


if __name__ == '__main__':
    from transformers import HfArgumentParser
    from arguments import ModelArguments, DataArguments
    from transformers import AutoTokenizer

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = '/home/klv/llama_hf/7B'
    data_args.dataset_name = 'record'
    data_args.refresh = True
    data_args.data_tag = 'debug'
    train_on_inputs = False
    data_args.data_max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)
    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)
    # test_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.test_split)



