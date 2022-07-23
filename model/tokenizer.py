"""
@File        :  tokenizer.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/17
@Version     :  1.0
@Description :  BertTokenizer with variable length of `[CLS]`
"""
from itertools import chain
from math import ceil, log, floor
from typing import Optional, List, Union, Dict, Tuple

from transformers import BertTokenizer
from transformers.file_utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding, EncodedInput


class LogTruncation:
    """ Find optimal truncated length using Newton's Method """

    def __init__(self, log_base: int, anchored_length: int = -1):
        self._log_base = log_base
        self._ln_base_inverse = 1 / log(log_base)
        self._anchored_length = anchored_length

    def _f(self, x: float, max_length: int) -> float:
        return x + log(x, self._log_base) - max_length

    def _f_derivative(self, x: float) -> float:
        return 1 + self._ln_base_inverse / x

    def get_prompt_len(self, len_ids: float, len_pair_ids: float) -> int:
        return ceil(log(max(1.5, len_ids + len_pair_ids), self._log_base))

    def truncated_length(
            self, len_ids: int, len_pair_ids: int = None, max_length: int = None
    ) -> Tuple[int, int, int]:
        """ :returns: prompt_len, len_ids, len_pair_ids """
        len_pair_ids = 0 if len_pair_ids is None else len_pair_ids
        rest = len_ids + len_pair_ids

        if self._anchored_length == -1:
            prompt_len = self.get_prompt_len(len_ids, len_pair_ids)

            if not (max_length is None or prompt_len + len_ids + len_pair_ids <= max_length):
                x = len_ids + len_pair_ids
                x = max(1.0, x - self._f(x, max_length) / self._f_derivative(x))
                x = max(1.0, x - self._f(x, max_length) / self._f_derivative(x))
                x = max(1.0, x - self._f(x, max_length) / self._f_derivative(x))

                prompt_len = self.get_prompt_len(x, 0)
                rest = max_length - prompt_len
        else:
            prompt_len = self._anchored_length
            if max_length:
                rest = max_length - prompt_len

        # 分配长度给ids和pair_ids
        if len_pair_ids == 0:
            len_ids = rest
        elif len_ids == 0:
            len_pair_ids = rest
        elif len_ids + len_pair_ids != rest:
            s = len_ids + len_pair_ids
            len_ids, len_pair_ids = floor(rest * len_ids / s), floor(rest * len_pair_ids / s)

        return prompt_len, len_ids, len_pair_ids


class LogLMTokenizer(BertTokenizer):
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask", "prompt_type_ids"]

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
            log_base=2,
            anchored_length: int = -1,
            **kwargs
    ):
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
            **kwargs,
        )
        self.prompt_tokens = [
            '[CLS]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]',
            '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]', '[unused11]'
        ]
        self.prompt_token_ids = [self._convert_token_to_id(t) for t in self.prompt_tokens]

        self.log_truncation = LogTruncation(log_base, anchored_length)

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

        required_input = encoded_inputs["prompt_type_ids"]
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if "prompt_type_ids" in encoded_inputs:
                    encoded_inputs["prompt_type_ids"] = encoded_inputs["prompt_type_ids"] + [0] * difference
            else:
                if "prompt_type_ids" in encoded_inputs:
                    encoded_inputs["prompt_type_ids"] = [0] * difference + encoded_inputs["prompt_type_ids"]

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
            return_prompt_type_ids: bool = True,
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
        prompt_len, _, _ = self.log_truncation.truncated_length(len_ids + 1,
                                                                len_pair_ids + 1 if len_pair_ids > 0 else 0,
                                                                max_length)
        total_len = prompt_len + len_ids + len_pair_ids

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
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

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

        if return_prompt_type_ids:
            encoded_inputs["prompt_type_ids"] = [1] * prompt_len + [0] * (len(sequence) - prompt_len)

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        prompt_len, _, _ = self.log_truncation.truncated_length(len(token_ids_0) + 1,
                                                                len(token_ids_1) + 1 if token_ids_1 else 0)
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return self.prompt_token_ids[:prompt_len] + token_ids_0 + sep
        else:
            return self.prompt_token_ids[:prompt_len] + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            sp_ids = getattr(self, "special_tokens_ids", None)
            if sp_ids is None:
                _sp_ids = [self._convert_token_to_id(t) for t in self.special_tokens_map.values()]
                sp_ids = set(_sp_ids + self.prompt_token_ids)
                setattr(self, "special_tokens_ids", sp_ids)
            return [1 if _id in sp_ids else 0 for _id in chain(token_ids_0, [] if token_ids_1 is None else token_ids_1)]

        prompt_len, _, _ = self.log_truncation.truncated_length(len(token_ids_0) + 1,
                                                                len(token_ids_1) + 1 if token_ids_1 else 0)

        if token_ids_1 is not None:
            return [1] * prompt_len + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        return [1] * prompt_len + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        prompt_len, _, _ = self.log_truncation.truncated_length(len(token_ids_0) + 1,
                                                                len(token_ids_1) + 1 if token_ids_1 else 0)

        if token_ids_1 is None:
            return [0] * (prompt_len + len(token_ids_0) + 1)

        return [0] * (prompt_len + len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)
