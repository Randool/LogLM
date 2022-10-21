"""
@File        :  dataset.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/18
@Version     :  1.0
"""
import random
import warnings
from collections import namedtuple
from math import floor, ceil
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset, Dataset

from model.tokenizer import Truncator, VCLSTokenizer

Item = namedtuple("Items", ("input_ids", "token_type_ids", "cls_type_ids", "labels", "SO_label"))


def str_to_int_array(string: str) -> List[int]:
    return list(map(int, string.split()))


def expand_masked_spans(masked_pos: str) -> List[int]:
    spans = masked_pos.split(" ")
    pos = []
    for span in spans:
        if "-" in span:
            start, end = span.split("-")
            pos.extend([i for i in range(int(start), 1 + int(end))])
        else:
            pos.append(int(span))
    return pos


def trim_ids(ids: List[Optional[int]], target_len: int, has_bos: bool, has_eos: bool, direct: str = "any") -> List[int]:
    delta = len(ids) - target_len
    if delta <= 0:
        return ids

    if direct == "left":
        for i in range(delta + has_bos):
            ids[i] = None

    elif direct == "right":
        for i in range(len(ids) - 1 - has_eos, 0, -1):
            ids[i] = None

    else:
        p, q = int(has_bos), len(ids) - 1 - has_eos

        while delta > 0:
            if random.random() < 0.5:
                ids[p] = None
                p += 1
            else:
                ids[q] = None
                q -= 1
            delta -= 1

    return [x for x in ids if x is not None]


def legacy_parse_example(line: str, truncator: Truncator, mask_token_id: int, max_length: int = None,
                         nar_mode: bool = True) -> Item:
    warnings.warn("`legacy_parse_example` is deprecated", DeprecationWarning)

    # Format: `labels|masked_pos|token_type_lengths|cls_length|SO_label` or `labels|masked_pos`
    labels_, masked_pos, token_type_lengths, SO_label = line.split("|")
    token_ids = str_to_int_array(labels_)
    masked_pos = expand_masked_spans(masked_pos)
    token_type_lengths = str_to_int_array(token_type_lengths)
    SO_label = int(SO_label)

    # Invert the position where mask is required
    for pos in masked_pos:
        token_ids[pos] = -token_ids[pos]

    cls_len, len_ids, len_pair_ids = truncator.truncated_length(token_type_lengths[0] - (not nar_mode),
                                                                token_type_lengths[1],
                                                                max_length)

    cls_ids = [i for i in range(1, cls_len + 1)]
    if nar_mode:
        ids = trim_ids(token_ids[:token_type_lengths[0]], len_ids, False, True)
        pair_ids = trim_ids(token_ids[token_type_lengths[0]:], len_pair_ids, False, True)
    else:
        ids = trim_ids(token_ids[:token_type_lengths[0] - 1], len_ids, False, False, direct="left")
        pair_ids = trim_ids(token_ids[token_type_lengths[0]:], len_pair_ids, False, True, direct="right")

    input_ids = cls_ids + [x if x > 0 else mask_token_id for x in ids] + [x if x > 0 else mask_token_id for x in
                                                                          pair_ids]
    labels = cls_ids + [abs(x) for x in ids] + [abs(x) for x in pair_ids]

    token_type_ids = [0] * (cls_len + len_ids) + [1] * len_pair_ids
    cls_type_ids = [1] * cls_len + [0] * (len_ids + len_pair_ids)

    return Item(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        cls_type_ids=cls_type_ids,
        labels=labels,
        SO_label=SO_label
    )


def parse_example(line: str, tokenizer: VCLSTokenizer, max_length: int = None,
                  cls_len_disturb_rate: float = 0.0) -> Item:
    """
    Parse example of T5-task format: `a b c d e f g|A-B C D-E`

    :param line: Example line.
    :param tokenizer:       VCLSTokenizer.
    :param max_length:      Max sequence length.
    :param cls_len_disturb_rate: If not 0, the length of CLS will increase or decrease randomly by this rate.
    """
    labels_, masked_pos = line.split("|")
    token_ids = str_to_int_array(labels_)
    masked_pos = expand_masked_spans(masked_pos)

    # Invert the position where mask is required
    for pos in masked_pos:
        token_ids[pos] = -token_ids[pos]

    cls_len, len_ids, _len_pair_ids = tokenizer.truncated_length(len(token_ids), max_length)

    if cls_len_disturb_rate > 0:
        cls_len = random.randint(
            max(1, floor(cls_len * (1 - cls_len_disturb_rate))),
            ceil(cls_len * (1 + cls_len_disturb_rate))
        )

    cls_ids = [tokenizer.cls_token_id] * cls_len
    ids = trim_ids(token_ids, len_ids, False, False, direct="any")

    input_ids = cls_ids + [x if x > 0 else tokenizer.mask_token_id for x in ids] + [tokenizer.eos_token_id]
    labels = cls_ids[:1] + [abs(x) for x in ids] + [tokenizer.eos_token_id]
    cls_type_ids = [1] * cls_len + [0] * (len_ids + 1)

    return Item(
        input_ids=input_ids,
        token_type_ids=None,
        cls_type_ids=cls_type_ids,
        labels=labels,
        SO_label=None
    )


def parse_example_t5(line: str, tokenizer: VCLSTokenizer, max_length: int = None,
                     cls_len_disturb_rate: float = 0.0) -> Item:
    """
    Parse example of T5-task format: `a b c d e f g|A-B C D-E`

    :param line: Example line.
    :param tokenizer:       VCLSTokenizer.
    :param max_length:      Max sequence length.
    :param cls_len_disturb_rate: If not 0, the length of CLS will increase or decrease randomly by this rate.
    """
    labels_, masked_pos = line.split("|")
    token_ids = str_to_int_array(labels_)
    masked_pos = expand_masked_spans(masked_pos)
    msk_idx = 0
    span_cnt = 0

    _input_ids = []

    # State is switch between in-masked-span and normal
    NORMAL, IN_MASK = 1, 2
    STATE = NORMAL
    t5_mask_token_ids = tokenizer.additional_special_tokens_ids

    for i, _id in enumerate(token_ids):
        if msk_idx < len(masked_pos) and i == masked_pos[msk_idx]:
            if STATE == NORMAL:
                _input_ids.append([t5_mask_token_ids[span_cnt], _id])
                span_cnt += 1
                STATE = IN_MASK
            else:
                _input_ids[-1].append(_id)
            msk_idx += 1
        else:
            if STATE == IN_MASK:
                STATE = NORMAL
            _input_ids.append(_id)

    cls_len, len_ids, _len_pair_ids = tokenizer.truncated_length(len(_input_ids), max_length)

    if cls_len_disturb_rate > 0:
        cls_len = random.randint(
            max(1, floor(cls_len * (1 - cls_len_disturb_rate))),
            ceil(cls_len * (1 + cls_len_disturb_rate))
        )

    cls_ids = [tokenizer.cls_token_id] * cls_len
    _input_ids = trim_ids(_input_ids, len_ids, False, False, direct="any")

    input_ids = cls_ids + [x if isinstance(x, int) else x[0] for x in _input_ids] + [tokenizer.eos_token_id]
    labels = cls_ids[:1] + [_x for x in _input_ids if isinstance(x, list) for _x in x] + [tokenizer.eos_token_id]
    cls_type_ids = [1] * cls_len + [0] * (len(input_ids) - cls_len)

    return Item(
        input_ids=input_ids,
        token_type_ids=None,
        cls_type_ids=cls_type_ids,
        labels=labels,
        SO_label=None
    )


class CollateFn:
    def __init__(self, mask_token_id: int, pad_token_id: int, nar_mode: bool, return_cls_ids: bool):
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.nar_mode = nar_mode  # Non-autoregressive mode
        self.return_cls_ids = return_cls_ids

    def __call__(self, batch: List[Item]) -> dict:
        input_ids, token_type_ids, cls_type_ids, labels, SO_label = [], [], [], [], []
        attention_mask = []
        max_length = 0
        label_max_length = 0
        is_legacy_format = batch[0].token_type_ids is not None

        for items in batch:
            input_ids.append(items.input_ids)
            if self.return_cls_ids:
                cls_type_ids.append(items.cls_type_ids)
            labels.append(items.labels)
            max_length = max(max_length, len(items.input_ids))
            label_max_length = max(label_max_length, len(items.labels))

            if is_legacy_format:
                token_type_ids.append(items.token_type_ids)
                SO_label.append(items.SO_label)

        for i in range(len(batch)):
            length = len(input_ids[i])
            pad_length = max_length - length
            label_pad_length = label_max_length - len(labels[i])
            input_ids[i] += [self.pad_token_id] * pad_length
            attention_mask.append([1] * length + [0] * pad_length)
            if self.return_cls_ids:
                cls_type_ids[i] += [0] * pad_length
            labels[i] += [-100] * label_pad_length if self.nar_mode else [self.pad_token_id] * label_pad_length

            if is_legacy_format:
                token_type_ids[i] += [0] * pad_length

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.return_cls_ids:
            inputs["cls_type_ids"] = torch.tensor(cls_type_ids, dtype=torch.long)

        if is_legacy_format and self.nar_mode:
            inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
            inputs["sentence_pair_label"] = torch.tensor(SO_label, dtype=torch.long)
            # inputs["mask_token_id"] = self.mask_token_id

        return inputs


class DatasetForShuffledFile(IterableDataset):
    def __init__(self, data_file: str, mask_token_id: int, pad_token_id: int, num_examples: int = None,
                 mapper: str = None, anchored_length: int = -1, nar_mode=True, return_cls_ids=True):
        self.data_file = data_file
        self.num_examples = num_examples
        self.nar_mode = nar_mode

        self._trunc = Truncator(mapper, anchored_length)
        self.collate_fn = CollateFn(mask_token_id, pad_token_id, nar_mode, return_cls_ids)

    def __iter__(self):
        with open(self.data_file, "r") as f:
            for line in f:
                if len(line) < 1:
                    continue
                yield legacy_parse_example(line, self._trunc, self.collate_fn.mask_token_id)

    def __len__(self):
        if self.num_examples:
            return self.num_examples
        warnings.warn(f"Counting `num_examples` on {self.data_file}, which is time consuming...")
        self.num_examples = 0
        with open(self.data_file) as f:
            for _ in f:
                self.num_examples += 1
        return self.num_examples


class DatasetWithIndex(Dataset):
    def __init__(self, data_file: str, tokenizer: VCLSTokenizer, max_length: int, mapper: str = None,
                 anchored_length: int = -1, nar_mode=False, t5_mode=False, return_cls_ids=True,
                 cls_len_disturb_rate=0.0, index_suffix: str = ".index"):
        self.data_file = data_file
        self.indexes = self.parse_index(data_file + index_suffix)

        self.max_length = max_length
        self.nar_mode = nar_mode
        self.t5_mode = t5_mode
        self.cls_len_disturb_rate = cls_len_disturb_rate

        self.tokenizer = tokenizer
        self.tokenizer.update_mapper_or_anchored_length(mapper, anchored_length)

        if t5_mode and tokenizer.additional_special_tokens_ids is None:
            raise ValueError("`t5_mask_token_ids` should be given under T5-mode")

        self.collate_fn = CollateFn(tokenizer.mask_token_id, tokenizer.pad_token_id, nar_mode, return_cls_ids)

    @staticmethod
    def parse_index(index_file: str) -> List[int]:
        with open(index_file, encoding="utf8") as f:
            indexes = [int(idx) for idx in f.read().split()]
        return indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        idx = self.indexes[index]

        with open(self.data_file, "rb") as f:
            f.seek(idx)
            line = f.readline().decode()

        if self.t5_mode:
            return parse_example_t5(line, tokenizer=self.tokenizer, max_length=self.max_length,
                                    cls_len_disturb_rate=self.cls_len_disturb_rate)

        return parse_example(line, tokenizer=self.tokenizer, max_length=self.max_length,
                             cls_len_disturb_rate=self.cls_len_disturb_rate)
