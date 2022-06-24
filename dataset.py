"""
@File        :  dataset.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/18
@Version     :  1.0
"""
import copy
import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


class DatasetForShuffledFile(IterableDataset):
    def __init__(self, data_file: str, mask_token_id: int, pad_token_id: int, num_examples: int = None):
        self.data_file = data_file
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self._num_examples = num_examples

    def __iter__(self):
        def str_to_int_array(string: str) -> List[int]:
            return list(map(int, string.split()))

        with open(self.data_file, "r") as f:
            for line in f:
                if len(line) < 1:
                    continue
                # Format: `labels|masked_pos|token_type_lengths|prompt_length|SO_label`
                labels, masked_pos, token_type_lengths, prompt_length, SO_label = line.split("|")

                labels = str_to_int_array(labels)
                masked_pos = str_to_int_array(masked_pos)
                token_type_lengths = str_to_int_array(token_type_lengths)
                prompt_length = int(prompt_length)
                SO_label = int(SO_label)

                # Recover `input_ids` from `labels` and `masked_pos`
                input_ids = copy.deepcopy(labels)
                for pos in masked_pos:
                    input_ids[pos] = self.mask_token_id

                token_type_ids = [0] * token_type_lengths[0] + [1] * token_type_lengths[1]
                prompt_type_ids = [1] * prompt_length + [0] * (len(labels) - prompt_length)

                yield input_ids, token_type_ids, prompt_type_ids, labels, SO_label

    def __len__(self):
        if self._num_examples:
            return self._num_examples
        with open(self.data_file) as f:
            for _ in f:
                self._num_examples += 1
        return self._num_examples

    def collate_fn(self, batch: List[tuple]) -> dict:
        # The model needs: input_ids, attention_mask, token_type_ids, prompt_type_ids, labels, sentence_pair_label
        input_ids, token_type_ids, prompt_type_ids, labels, SO_label = [], [], [], [], []
        attention_mask = []
        max_length = 0

        for pair in batch:
            input_ids.append(pair[0])
            token_type_ids.append(pair[1])
            prompt_type_ids.append(pair[2])
            labels.append(pair[3])
            SO_label.append(pair[4])
            max_length = max(max_length, len(pair[0]))

        for i in range(len(batch)):
            length = len(input_ids[i])
            pad_length = max_length - length
            input_ids[i] += [self.pad_token_id] * pad_length
            attention_mask.append([1] * length + [0] * pad_length)
            token_type_ids[i] += [0] * pad_length
            prompt_type_ids[i] += [0] * pad_length
            labels[i] += [self.pad_token_id] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "prompt_type_ids": torch.tensor(prompt_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sentence_pair_label": torch.tensor(SO_label, dtype=torch.long),
            "mask_token_id": self.mask_token_id,
        }


class SkipDataset(Dataset):
    """
    NOTE: When skipping data, return None to escape the processing of collator function.
    """

    def __init__(self, skipNum: int = 0) -> None:
        super().__init__()
        self.skipNum = skipNum

    def setSkipNum(self, skipNum: int):
        self.skipNum = skipNum

    def clearSkipNum(self):
        self.skipNum = 0


class JsonLineLargeDataset(SkipDataset):
    """
    The dataset will open the large file when we read one sample.
    """

    def __init__(self, dataFile: str = None, indexFile: str = None) -> None:
        super().__init__()

        self.dataFile = dataFile
        self.offsets = self._loadIndex(indexFile)

    @staticmethod
    def _loadIndex(indexFile: str) -> List[int]:
        with open(indexFile, "r") as reader:
            indices = list(map(int, reader.read().strip().split("\n")))
        return np.array(indices, dtype=int)

    def __getitem__(self, index: int) -> Dict:
        if self.skipNum > 0:
            self.skipNum -= 1
            return None
        byteOffset = self.offsets[index]

        # NOTE: *** open the data file every time ***
        with open(self.dataFile, "rb") as reader:
            reader.seek(byteOffset, os.SEEK_SET)
            line = reader.readline().decode().strip()
            example = json.loads(line)

        return example

    def __len__(self) -> int:
        return len(self.offsets)


def MLMCollateFunc(batch: List, padTokenId: int, clsTokenId: int = None) -> Dict:
    inputs = {
        "input_ids": [],
        "token_type_ids": [],
        "tokenLabels": [],
        "sentLabels": [],
        "word_left_bound": [],
        "word_right_bound": [],
    }

    # gather
    for example in batch:
        inputs["input_ids"].append(example["MaskedToken"])
        inputs["token_type_ids"].append(example["segBound"])
        inputs["tokenLabels"].append(example["MaskedLabel"])
        inputs["sentLabels"].append(example["SOLabel"])
        inputs["word_left_bound"].append(example["wordLeftBound"])
        inputs["word_right_bound"].append(example["wordRightBound"])

    # transform to Tensors
    inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
    inputs["tokenLabels"] = torch.LongTensor(inputs["tokenLabels"])
    inputs["sentLabels"] = torch.LongTensor(inputs["sentLabels"])
    inputs["word_left_bound"] = torch.LongTensor(inputs["word_left_bound"])
    inputs["word_right_bound"] = torch.LongTensor(inputs["word_right_bound"])

    # token type ids
    maxLen = inputs["input_ids"].size(1)
    boundary = torch.LongTensor(inputs["token_type_ids"]).view(-1, 1)
    order = torch.arange(maxLen, dtype=torch.long)
    token_type_ids = torch.ge(order, boundary).type(torch.long)
    inputs["token_type_ids"] = token_type_ids

    # attention mask
    inputs["attention_mask"] = torch.ne(inputs["input_ids"], padTokenId).type(torch.int)

    return inputs


def DNCollateFunc(batch: List, padTokenId: int, clsTokenId: int) -> Dict:
    inputs = {
        "input_ids": [],
        "dn_decoder_input_ids": [],
        "dnLabels": [],
    }

    for example in batch:
        inputs["input_ids"].append(example["sharedToken"])
        inputs["dn_decoder_input_ids"].append([clsTokenId] + example["dnLabel"][:-1])
        inputs["dnLabels"].append(example["dnLabel"])

    inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
    inputs["dn_decoder_input_ids"] = torch.LongTensor(inputs["dn_decoder_input_ids"])
    inputs["dn_decoder_input_ids"][inputs["dn_decoder_input_ids"] == -100] = padTokenId
    inputs["dnLabels"] = torch.LongTensor(inputs["dnLabels"])

    inputs["attention_mask"] = torch.ne(inputs["input_ids"], padTokenId).type(torch.int)

    return inputs


def MLMWithDNCollateFunc(batch: List, padTokenId: int, clsTokenId: int) -> Dict:
    inputs = MLMCollateFunc(batch, padTokenId)
    inputs["dn_decoder_input_ids"] = []
    inputs["dnLabels"] = []

    for example in batch:
        inputs["dn_decoder_input_ids"].append([clsTokenId] + example["dnLabel"][:-1])
        inputs["dnLabels"].append(example["dnLabel"])

    inputs["dn_decoder_input_ids"] = torch.LongTensor(inputs["dn_decoder_input_ids"])
    inputs["dn_decoder_input_ids"][inputs["dn_decoder_input_ids"] == -100] = padTokenId
    inputs["dnLabels"] = torch.LongTensor(inputs["dnLabels"])

    return inputs
