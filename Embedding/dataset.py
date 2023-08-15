# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from io_util import load_jsonl


def get_field_names(data_item):
    if "text1" in data_item and "text2" in data_item:
        return "text1", "text2"
    elif "sentence1" in data_item and "sentence2" in data_item:
        return "sentence1", "sentence2"
    else:
        return None, None


def load_cosent_train_data(dataset_dir):
    data = []

    file_list = os.listdir(dataset_dir)

    for fil in file_list:
        if fil.startswith('train'):
            path = os.path.join(dataset_dir, fil)
            if path.endswith('.jsonl'):
                data_list = load_jsonl(path)
                for entry in data_list:
                    field1, field2 = get_field_names(entry)
                    if not field1 or not field2:
                        continue

                    text_a, text_b, score = entry[field1], entry[field2], float(entry["label"])
                    data.append((text_a, score))
                    data.append((text_b, score))
            else:
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        if len(line) != 3:
                            logger.warning(f'line size not match, pass: {line}')
                            continue
                        score = float(line[2])
                        data.append((line[0], score))
                        data.append((line[1], score))

            logger.info(f"load train file: {fil}\n")
    return data


class CosentTrainDataset(Dataset):
    """Cosent文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), line[1]


def load_text_matching_test_data(dataset_dir):
    """
    Load test data from file.
        args: file path
        return: list of (text_a, text_b, score)
    """
    data = []

    file_list = os.listdir(dataset_dir)
    for fil in file_list:
        if fil.startswith('eval'):
            path = os.path.join(dataset_dir, fil)

            if path.endswith('.jsonl'):
                data_list = load_jsonl(path)
                for entry in data_list:
                    field1, field2 = get_field_names(entry)
                    if not field1 or not field2:
                        continue

                    text_a, text_b, score = entry[field1], entry[field2], int(entry["label"])
                    data.append((text_a, text_b, score))
            else:
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        line = line.strip().split('\t')
                        if len(line) != 3:
                            logger.warning(f'line size not match, pass: {line}')
                            continue
                        score = int(line[2])
                        data.append((line[0], line[1], score))
            logger.info(f"load eval file: {fil}\n")
    return data


class TextMatchingTrainDataset(Dataset):
    """文本匹配训练数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]


class TextMatchingTestDataset(Dataset):
    """文本匹配测试数据集, 重写__getitem__和__len__方法"""

    def __init__(self, tokenizer: PreTrainedTokenizer, data: list, max_len: int = 512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return self.tokenizer(text, max_length=self.max_len, truncation=True,
                              padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        line = self.data[index]
        return self.text_2_id(line[0]), self.text_2_id(line[1]), line[2]
