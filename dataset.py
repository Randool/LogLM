"""
@File        :  dataset.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/18
@Version     :  1.0
"""
import warnings
from collections import namedtuple
from typing import List

import torch
from torch.utils.data import IterableDataset

from model.tokenizer import LogTruncation

Items = namedtuple("Items", ("input_ids", "token_type_ids", "cls_type_ids", "labels", "SO_label"))


class DatasetForShuffledFile(IterableDataset):
    def __init__(self, data_file: str, mask_token_id: int, pad_token_id: int, num_examples: int = None, log_base=2):
        self.data_file = data_file
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.num_examples = num_examples

        self._trunc = LogTruncation(log_base=log_base)

    def __iter__(self):
        def str_to_int_array(string: str) -> List[int]:
            return list(map(int, string.split()))

        with open(self.data_file, "r") as f:
            for line in f:
                if len(line) < 1:
                    continue
                # Format: `labels|masked_pos|token_type_lengths|cls_length|SO_label`
                labels, masked_pos, token_type_lengths, cls_length, SO_label = line.split("|")

                labels = str_to_int_array(labels)
                masked_pos = str_to_int_array(masked_pos)
                token_type_lengths = str_to_int_array(token_type_lengths)
                cls_length = int(cls_length)
                SO_label = int(SO_label)

                # Remove or add some of cls tokens
                cls_ids, token_ids = labels[:cls_length], labels[cls_length:]
                new_cls_length = self._trunc.truncated_length(len_ids=len(token_ids))[0]
                assert new_cls_length <= 12, f"new_cls_length ({new_cls_length}) > 12"

                while new_cls_length > len(cls_ids):
                    cls_ids.append(cls_ids[-1] + 1)

                labels = cls_ids[:new_cls_length] + token_ids
                mask_pos_bias = new_cls_length - cls_length
                cls_length = new_cls_length

                token_type_lengths[0] += mask_pos_bias

                # Recover `input_ids` from `labels` and `masked_pos`
                input_ids = [id_ for id_ in labels]
                for pos in masked_pos:
                    input_ids[pos + mask_pos_bias] = self.mask_token_id

                token_type_ids = [0] * token_type_lengths[0] + [1] * token_type_lengths[1]
                cls_type_ids = [1] * cls_length + [0] * (len(labels) - cls_length)

                yield Items(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    cls_type_ids=cls_type_ids,
                    labels=labels,
                    SO_label=SO_label
                )

    def __len__(self):
        if self.num_examples:
            return self.num_examples
        warnings.warn(f"Counting `num_examples` on {self.data_file}, which is time consuming...")
        self.num_examples = 0
        with open(self.data_file) as f:
            for _ in f:
                self.num_examples += 1
        return self.num_examples

    def collate_fn(self, batch: List[Items]) -> dict:
        input_ids, token_type_ids, cls_type_ids, labels, SO_label = [], [], [], [], []
        attention_mask = []
        max_length = 0

        for items in batch:
            input_ids.append(items.input_ids)
            token_type_ids.append(items.token_type_ids)
            cls_type_ids.append(items.cls_type_ids)
            labels.append(items.labels)
            SO_label.append(items.SO_label)
            max_length = max(max_length, len(items.input_ids))

        for i in range(len(batch)):
            length = len(input_ids[i])
            pad_length = max_length - length
            input_ids[i] += [self.pad_token_id] * pad_length
            attention_mask.append([1] * length + [0] * pad_length)
            token_type_ids[i] += [0] * pad_length
            cls_type_ids[i] += [0] * pad_length
            labels[i] += [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "cls_type_ids": torch.tensor(cls_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sentence_pair_label": torch.tensor(SO_label, dtype=torch.long),
            "mask_token_id": self.mask_token_id,
        }
