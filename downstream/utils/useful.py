"""
@File        :  useful.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/3/22
@Version     :  1.0
@Description :  Useful functions
"""
import copy
import glob
import json
import os
from typing import List, Any, Tuple, Union, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer, HfArgumentParser

from utils import get_logger


def get_logger_and_args(logger_name: str, args_parsers: list, _args: List[str] = None):
    """
    Args:
        logger_name:    Logger name
        args_parsers:   Arguments parsers defined in src/utils/args.py. `train_args` should be the first one.
        _args:          Arguments
    """
    parser = HfArgumentParser(args_parsers)
    arg_namespaces = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=arg_namespaces[0].logging_dir, rank=arg_namespaces[0].local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    for arg_ns in arg_namespaces:
        logger.info(str(arg_ns.to_json_string()))

    return logger, arg_namespaces


def best_model_checkpoint(output_dir: str) -> str:
    """ Select best model checkpoint """
    trainer_states = glob.glob(os.path.join(output_dir, "checkpoint-*/trainer_state.json"))

    largest = 0
    best_path = None

    for state in trainer_states:
        with open(state, encoding="utf8") as f:
            path = json.load(f)["best_model_checkpoint"]
        steps = int(path.split("-")[-1])
        if steps > largest:
            largest = steps
            best_path = path

    return best_path


def extend_special_tokens_for_tokenizer(tokenizer: PreTrainedTokenizer, special_tokens: List[str]):
    """ Add special customized tokens like: <DOC>, <PAT>, <ACT>, ... """
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    # if (tokenizer.bos_token is None) and ("cls_token" in tokenizer.special_tokens_map):
    #     special_tokens_dict["bos_token"] = tokenizer.cls_token
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def inputs_padding(
        input_ids: List[List[int]], pad_token_id: int, padding_to_len: int = -1, add_attention: bool = True,
        labels: List[List[int]] = None, label_pad_token_id: int = None
) -> dict:
    """ Padding for input_ids, attention_mask and labels. Used in collate_fn """
    max_len = max(max(map(len, input_ids)), padding_to_len)
    attention_mask = torch.zeros((len(input_ids), max_len), dtype=torch.long) if add_attention else None

    input_ids = copy.deepcopy(input_ids)
    labels = copy.deepcopy(labels)

    for i, _ids in enumerate(input_ids):
        if add_attention:
            attention_mask[i][:len(_ids)] = 1
        _delta_len = max_len - len(_ids)
        input_ids[i] += [pad_token_id] * _delta_len
        if labels is not None:
            labels[i] += [label_pad_token_id] * _delta_len

    inputs = {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
    if add_attention:
        inputs["attention_mask"] = attention_mask
    if labels is not None:
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs


def inputs_to_device(inputs, device):
    if isinstance(inputs, (dict, BatchEncoding)):
        return {k: inputs_to_device(v, device) for k, v in inputs.items()}

    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(inputs_to_device(x, device) for x in inputs)

    elif torch.is_tensor(inputs):
        return inputs.to(device)

    return inputs


def KMP_search(pattern: Union[str, List[Any]], target: Union[str, List[Any]], max_match=-1) -> List[Tuple[int, int]]:
    """ 执行KMP搜索，返回左闭右开区间列表 """
    _next = [0] * len(pattern)

    for i in range(1, len(pattern)):
        j = _next[i - 1]
        while j > 0 and pattern[j] != pattern[i]:
            j = _next[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        _next[i] = j

    matched_pos = []
    j = 0

    for i in range(len(target)):
        while j > 0 and target[i] != pattern[j]:
            j = _next[j - 1]
        if target[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            matched_pos.append((i - j + 1, i + 1))
            if 0 < max_match <= len(matched_pos):
                break
            j = _next[j - 1]

    return matched_pos


def k_fold_recombine(trainset: Dataset, devset: Dataset, swap_attrs: List[str], curr_fold: int,
                     total_folds: int) -> Tuple[Dataset, Dataset]:
    """
    重组K折交叉验证所需的训练和验证数据

    Args:
        trainset:       原始训练集
        devset:         原始验证集
        swap_attrs:     需要做k折交换的attr，比如data、examples这种，这些attr必须是`list`或`tuple`
        curr_fold:      当前编号，0 <= curr_fold < total_folds
        total_folds:    总编号数量，以此为模数
    Return:
        重组数据后的trainset和devset
    """
    assert 0 <= curr_fold < total_folds, f"curr_fold: {curr_fold}, total_fold: {total_folds}"

    if total_folds <= 1:
        return trainset, devset

    for attr in swap_attrs:
        if getattr(trainset, attr, None) is None:
            raise ValueError(f"{attr} not in trainset")
        if getattr(devset, attr, None) is None:
            raise ValueError(f"{attr} not in devset")

        buff = getattr(trainset, attr) + getattr(devset, attr)
        _type = type(getattr(trainset, attr))

        # 重巡分配
        new_train_attr = _type([x for i, x in enumerate(buff) if i % total_folds != curr_fold])
        new_dev_attr = _type([x for i, x in enumerate(buff) if i % total_folds == curr_fold])
        setattr(trainset, attr, new_train_attr)
        setattr(devset, attr, new_dev_attr)

    return trainset, devset


# 一些基于numpy的函数
def topk_element_in_ndarray(matrix: np.ndarray, topk: int):
    """ 在矩阵中寻找topk元素（无序），同时返回坐标 """
    coo: Tuple[np.ndarray] = np.unravel_index(np.argpartition(matrix.ravel(), -topk)[-topk:], matrix.shape)
    val: np.ndarray = matrix[coo]
    return val, coo


def argmax(iterable: Iterable, key=lambda x: x):
    return max(enumerate(iterable), key=lambda x: key(x[1]))[0]


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    _max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - _max)
    _sum = np.sum(e_x, axis=-1, keepdims=True)
    f_x = e_x / _sum
    return f_x


def max_product(matrix1, matrix2):
    """ 自定义矩阵乘法：M = M1 @ M2, M[i,j] = max(M1[i,k] + M2[k,j]) """
    d1, d2 = matrix1.shape
    d3, d4 = matrix2.shape
    assert d2 == d3
    matrix = np.zeros((d1, d4))
    matrix.fill(-np.inf)
    for i in range(d1):
        for j in range(d4):
            matrix[i, j] = max(matrix1[i, :] + matrix2[:, j])
    return matrix
