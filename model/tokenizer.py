"""
@File        :  tokenizer.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/17
@Version     :  1.0
@Description :  BertTokenizer with variable length of `[CLS]`
"""
import json
import os
from contextlib import contextmanager
from functools import lru_cache
from itertools import chain
from math import ceil, floor
from typing import Optional, List, Union, Dict, Tuple

import sympy
from transformers import BertTokenizer
from transformers.file_utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, EncodedInput, TOKENIZER_CONFIG_FILE

DEFAULT_MAPPER = 'log(x,2)'


class Truncator:
    mapper: sympy.Expr
    _dy: sympy.Expr
    _x: sympy.Symbol
    anchored_length: int

    def __init__(self, mapper: str = None, anchored_length: int = -1):
        mapper = DEFAULT_MAPPER if mapper is None else mapper
        self.update_mapper_or_anchored_length(mapper, anchored_length)

    def update_mapper_or_anchored_length(self, mapper: str = None, anchored_length: int = None):
        if mapper is not None:
            mapper = sympy.parse_expr(mapper)

            assert len(mapper.free_symbols) == 1, \
                f'`mapper` must contain only one free symbol, got: {mapper.free_symbols}'

            self._x: sympy.Symbol = list(mapper.free_symbols)[0]
            self.mapper = mapper
            self._dy = sympy.diff(self.mapper + self._x)

        if anchored_length is not None:
            self.anchored_length = anchored_length

    def get_cls_len(self, len_ids: float, len_pair_ids: float = 0) -> int:
        return ceil(self.mapper.evalf(subs={self._x.name: len_ids + len_pair_ids}))

    @lru_cache(maxsize=128)
    def truncated_length(self, len_ids: int, len_pair_ids: int = 0, max_length: int = 100000) -> Tuple[int, int, int]:
        """ :returns: cls_len, len_ids, len_pair_ids """
        rest = len_ids + len_pair_ids

        if self.anchored_length == -1:
            cls_len = self.get_cls_len(len_ids, len_pair_ids)

            if not (cls_len + len_ids + len_pair_ids <= max_length):
                f = self._x + self.mapper - max_length
                x_ = len_ids + len_pair_ids
                subs = {self._x.name: x_}

                for _ in range(3):
                    subs[self._x.name] = x_
                    f_ = float(f.evalf(subs=subs))
                    df_ = float(self._dy.evalf(subs=subs))
                    x_ = x_ - f_ / df_

                cls_len = self.get_cls_len(x_, 0)
                rest = max_length - cls_len

        else:
            cls_len = self.anchored_length
            if max_length:
                rest = max_length - cls_len

        # 分配长度给ids和pair_ids
        if len_pair_ids == 0:
            len_ids = rest
        elif len_ids == 0:
            len_pair_ids = rest
        elif len_ids + len_pair_ids != rest:
            s = len_ids + len_pair_ids
            if len_ids < len_pair_ids:
                len_pair_ids = floor(rest * len_pair_ids / s)
                len_ids = rest - len_pair_ids
            else:
                len_ids = floor(rest * len_ids / s)
                len_pair_ids = rest - len_ids

        return cls_len, len_ids, len_pair_ids


class VCLSTokenizer(BertTokenizer):
    model_input_names: List[str] = ["input_ids", "attention_mask", "cls_type_ids"]

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            mapper: str = DEFAULT_MAPPER,
            anchored_length: int = -1,
            return_token_type=False,
            extra_ids=100,
            additional_special_tokens=None,
            **kwargs
    ):
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.bos_token = self.cls_token
        self.eos_token = self.sep_token

        self.return_token_type = return_token_type
        if return_token_type:
            self.model_input_names.append("token_type_ids")

        self.truncator = Truncator(mapper, anchored_length)

    def update_mapper_or_anchored_length(self, mapper: sympy.Expr = None, anchored_length: int = None):
        self.truncator.update_mapper_or_anchored_length(mapper, anchored_length)

    @property
    def anchored_length(self):
        return self.truncator.anchored_length

    def truncated_length(self, len_ids: int, len_pair_ids: int = 0, max_length: int = 100000) -> Tuple[int, int, int]:
        return self.truncator.truncated_length(len_ids, len_pair_ids, max_length)

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None
    ):
        encoded_inputs = super()._pad(
            encoded_inputs, max_length=max_length, padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of, return_attention_mask=return_attention_mask
        )

        if "cls_type_ids" in encoded_inputs:
            required_input = encoded_inputs["cls_type_ids"]
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

            if needs_to_be_padded:
                difference = max_length - len(required_input)

                if self.padding_side == "right":
                    if "cls_type_ids" in encoded_inputs:
                        encoded_inputs["cls_type_ids"] = encoded_inputs["cls_type_ids"] + [0] * difference
                else:
                    if "cls_type_ids" in encoded_inputs:
                        encoded_inputs["cls_type_ids"] = [0] * difference + encoded_inputs["cls_type_ids"]

        return encoded_inputs

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            return_cls_type_ids: bool = True,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
    ) -> BatchEncoding:
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        cls_len, _, _ = self.truncator.truncated_length(len_ids + 1,
                                                        len_pair_ids + 1 if len_pair_ids > 0 else 0,
                                                        max_length)
        total_len = cls_len + len_ids + len_pair_ids

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            num_tokens_to_remove = total_len - max_length + 1  # [SEP]
            if pair_ids:
                num_tokens_to_remove += 1

            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=num_tokens_to_remove,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids) if return_token_type_ids else None
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else []) if return_token_type_ids else None

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        if return_cls_type_ids:
            encoded_inputs["cls_type_ids"] = [1] * cls_len + [0] * (len(sequence) - cls_len)

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls_len, *_ = self.truncator.truncated_length(len(token_ids_0) + 1, len(token_ids_1) + 1 if token_ids_1 else 0)
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return [self.cls_token_id] * cls_len + token_ids_0 + sep
        else:
            return [self.cls_token_id] * cls_len + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            sp_ids = getattr(self, "special_tokens_ids", None)
            if sp_ids is None:
                sp_ids = set(
                    [self._convert_token_to_id(t) for t in self.special_tokens_map.values()] + [self.cls_token_id]
                )
                setattr(self, "special_tokens_ids", sp_ids)
            return [1 if _id in sp_ids else 0 for _id in chain(token_ids_0, [] if token_ids_1 is None else token_ids_1)]

        cls_len, *_ = self.truncator.truncated_length(len(token_ids_0) + 1, len(token_ids_1) + 1 if token_ids_1 else 0)

        if token_ids_1 is not None:
            return [1] * cls_len + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        return [1] * cls_len + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls_len, *_ = self.truncator.truncated_length(len(token_ids_0) + 1, len(token_ids_1) + 1 if token_ids_1 else 0)

        if token_ids_1 is None:
            return [0] * (cls_len + len(token_ids_0) + 1)

        return [0] * (cls_len + len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            legacy_format: Optional[bool] = None,
            filename_prefix: Optional[str] = None,
            push_to_hub: bool = False,
            **kwargs,
    ) -> Tuple[str]:
        returned = super().save_pretrained(
            save_directory=save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub
        )

        # Add `log_base` & `anchored_length` to `tokenizer_config.json`
        with open(os.path.join(save_directory, TOKENIZER_CONFIG_FILE), "r+", encoding="utf8") as f:
            config = json.load(f)
            config["mapper"] = str(self.truncator.mapper)
            config["anchored_length"] = self.anchored_length
            config["return_token_type"] = self.return_token_type
            f.seek(0)
            json.dump(config, f)

        return returned

    @contextmanager
    def as_target_tokenizer(self):
        anchored_length = self.anchored_length
        self.update_mapper_or_anchored_length(anchored_length=1)
        yield
        self.update_mapper_or_anchored_length(anchored_length=anchored_length)
