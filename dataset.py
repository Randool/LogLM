"""
@File        :  dataset.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/18
@Version     :  1.0
"""
import json
import os
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetForPreTrain(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


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
