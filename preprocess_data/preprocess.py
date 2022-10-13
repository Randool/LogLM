"""
Author: Da Ma
Date: 2021.12.31
Description: Some preprocessor of transforming normalized text data into tokenized pre-training data.

REFER TO: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L338
"""
import argparse
import copy
import multiprocessing
import os
import random
import traceback
from multiprocessing import Pool
from os.path import join
from typing import List, Tuple, TextIO, Union, Iterable

import thulac

from model_x import LogLMTokenizer
from utils.logger import get_logger


class Config:
    def __init__(self, vocabPath: str, **kwargs) -> None:
        """Define several setting parameters."""
        self.vocabPath = vocabPath

        self.maskRatio = kwargs.get("maskRatio", 0.15)
        self.maskProb = kwargs.get("maskProb", 0.8)

        self.max_extension_len = kwargs.get("max_extension_len", 8)

        self.useWWM = kwargs.get("useWWM", False)
        self.useSynonym = kwargs.get("useSynonym", False)

        self.maxSeqLen = kwargs.get("maxSeqLen", 512)
        self.maxPredPerSeq = kwargs.get("maxPredPerSeq", 76)

        self.maxSeqLen2 = kwargs.get("maxSeqLen2", 4096)
        self.maxPredPerSeq2 = kwargs.get("maxPredPerSeq2", 256)

        self.extraSegWordsFile: str = kwargs.get("extraSegWordsFile", None)
        self.verboseStep: int = kwargs.get("verboseStep", None)
        self.NAR: bool = kwargs.get("NAR", True)


class Preprocessor:
    input_file: str
    writer: TextIO
    loggerPrefix: str

    def __init__(self, config_: Config, log_path: str = None):
        self.config = config_
        self.logger = get_logger(__name__, log_path)
        self.tokenizer = LogLMTokenizer(vocab_file=self.config.vocabPath)
        self.segmentor = thulac.thulac(user_dict=self.config.extraSegWordsFile, seg_only=True)

    @staticmethod
    def int_array_to_str(array: Iterable[int]) -> str:
        return " ".join(map(str, array))

    @staticmethod
    def is_CJK_piece(cp: str):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all the other languages.
        for p in cp:
            _ord = ord(p)
            if not (
                    (0x4E00 <= _ord <= 0x9FFF)
                    or (0x3400 <= _ord <= 0x4DBF)  #
                    or (0x20000 <= _ord <= 0x2A6DF)  #
                    or (0x2A700 <= _ord <= 0x2B73F)  #
                    or (0x2B740 <= _ord <= 0x2B81F)  #
                    or (0x2B820 <= _ord <= 0x2CEAF)  #
                    or (0xF900 <= _ord <= 0xFAFF)
                    or (0x2F800 <= _ord <= 0x2FA1F)  #
            ):  #
                return False

        return True

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()  # lowercase
        text = text.replace("#", " ")  # remote all the '#' character
        return text.strip()

    @staticmethod
    def truncate_sent_pair(tokens_a: List[str], tokens_b: List[str], max_num_tokens: int):
        """Truncates a pair of sequences to a maximum sequence length."""
        while len(tokens_a) + len(tokens_b) > max_num_tokens:
            choose_a = len(tokens_a) > len(tokens_b)
            trunc_tokens = tokens_a if choose_a else tokens_b
            assert len(trunc_tokens) >= 1

            if choose_a:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    @staticmethod
    def truncate_sent(tokens: List[str], max_num_tokens: int) -> None:
        while len(tokens) > max_num_tokens:
            if random.random() < 0.5:
                del tokens[0]
            else:
                tokens.pop()

    @staticmethod
    def convert_tokens_to_string(tokens: List[str]) -> str:
        """ ['化', '学', '药', '品', '：', 'aaa', '##a', 'bb', '##b']
        => '化学药品：aaaa bbb' """
        new_tokens = []
        for token in tokens:
            if MLMPreprocessor.is_CJK_piece(token):
                new_tokens.append(token)
            elif str.isalnum(token):
                new_tokens.append(" ")
                new_tokens.append(token)
            elif token.startswith("##"):
                new_tokens.append(token[2:])
            else:
                new_tokens.append(token)
        return "".join(new_tokens)

    def do_align(self, tokens: List[str]) -> List[List[str]]:
        """ ['化', '学', '药', '品', '：', 'aaa', '##a', 'bb', '##b']
        => [["化", "学", "药", "品"], ["aaa", "##a"], ["bb", "#b"]] """
        string = self.convert_tokens_to_string(tokens)  # 化学药品：aaaa bbb
        words = self.segmentor.cut(string, text=True).split()  # ["化学药品", "：", "aaaabbb"]
        token_idx = 0
        word_idx = 0
        pos_in_word = 0
        segments = []

        while word_idx < len(words):
            word = words[word_idx]
            segments.append([])

            while pos_in_word < len(word):
                token_ = tokens[token_idx]
                if token_.startswith("##"):
                    token_ = token_[2:]

                segments[-1].append(tokens[token_idx])
                pos_in_word += len(token_)
                token_idx += 1

            # 当word的切割粒度比token更细时会溢出
            if pos_in_word > len(word):
                while word_idx < len(words) and pos_in_word >= len(words[word_idx]):
                    pos_in_word -= len(words[word_idx])
                    word_idx += 1
            else:
                pos_in_word = 0
                word_idx += 1

        return segments

    def masked_pos(self, token_ids: Union[List[int], List[List[int]]], bias: int) -> List[str]:
        """ Just save the locations of the masked token. The "reservoir sampling"
            algorithm is used to limit the maximum number of predictions """
        mask_ratio = self.config.maskRatio
        max_pred_len = self.config.maxPredPerSeq
        pos = []

        if isinstance(token_ids[0], int):
            for i in range(len(token_ids)):
                if random.random() <= mask_ratio:
                    if len(pos) < max_pred_len:
                        pos.append(i + bias)
                    else:
                        replaced_idx = random.randint(0, i)
                        if replaced_idx < max_pred_len:
                            pos[replaced_idx] = i + bias
        else:
            chunk_pos = []
            for i, ids in enumerate(token_ids):  # type: int, List[int]
                if random.random() <= mask_ratio:
                    chunk = [j + bias for j in range(len(ids))]

                    if len(chunk_pos) < max_pred_len:
                        chunk_pos.append(chunk)
                    else:
                        replaced_idx = random.randint(0, i)
                        if replaced_idx < max_pred_len:
                            chunk_pos[replaced_idx] = chunk

                bias += len(ids)
            pos = [idx for chunk in chunk_pos for idx in chunk]

        # Merge spans
        pos = sorted(pos)
        spans = []
        i = 0

        while i < len(pos):
            j = i + 1
            while j < len(pos) and pos[j - 1] + 1 == pos[j]:
                j += 1

            if i + 1 == j:
                spans.append(str(pos[i]))
            else:
                spans.append(f"{pos[i]}-{pos[j - 1]}")
            i = j

        return spans

    def _create_sample(self, *args, **kwargs):
        raise NotImplementedError

    def _save_sample_to_output_file(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _create_and_save_instances_from_document(self, document: List[List[str]]) -> None:
        raise NotImplementedError

    def _create_training_instances_for_single_file(self) -> None:
        """
        Input file format:
            (1) One sentence per line. These should ideally be actual sentences, not
            entire paragraphs or arbitrary spans of text. (Because we use the
            sentence boundaries for the "next sentence prediction" task).
            (2) Blank lines between documents. Document boundaries are needed so
            that the "next sentence prediction" task doesn't span between documents.
        """
        document: List[List[str]] = []
        docNum = 0

        with open(self.input_file, "r") as reader:
            while True:
                line = reader.readline().strip()
                line = self.clean_text(line)

                # Empty lines are used as document delimiters
                if not line:
                    if not document:
                        break

                    # create instances for the document
                    self._create_and_save_instances_from_document(document)
                    docNum += 1

                    if (
                            self.config.verboseStep is not None
                            and not (docNum % self.config.verboseStep)
                            and self.logger is not None
                    ):
                        self.logger.info(f"{self.loggerPrefix}finish processing {docNum} documents!")

                    document = []

                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    document.append(tokens)

    def run(self, input_file: str, output_file: str, epoch: int):
        self.input_file = input_file
        self.writer = open(output_file, "w")
        self.loggerPrefix = f"epoch {epoch}, file {os.path.basename(input_file)} ===> "
        self._create_training_instances_for_single_file()
        self.writer.close()


class MLMPreprocessor(Preprocessor):
    def __init__(self, config_: Config, log_path: str = None):
        super().__init__(config_, log_path)

    def _create_sample(self, tokens_a: List[str], tokens_b: List[str]):
        token_type_lengths = (len(tokens_a) + 1, len(tokens_b) + 1)

        # cls_token_ids = self.tokenizer.cls_token_ids[:cls_len]
        sep_token_id = [self.tokenizer.sep_token_id]

        input_ids_a: List[int] = self.tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b: List[int] = self.tokenizer.convert_tokens_to_ids(tokens_b)
        labels = input_ids_a + sep_token_id + input_ids_b + sep_token_id

        if self.config.useWWM:
            aligned_tokens_a = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens_a)]
            aligned_tokens_b = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens_b)]
            masked_pos = self.masked_pos(aligned_tokens_a, 0) + \
                self.masked_pos(aligned_tokens_b, 0 + len(input_ids_a) + 1)
        else:
            masked_pos = self.masked_pos(input_ids_a, 0) + \
                self.masked_pos(input_ids_b, 0 + len(input_ids_a) + 1)

        return labels, masked_pos, token_type_lengths

    def _save_sample_to_output_file(self, labels: List[int], masked_pos: List[str],
                                    token_type_lengths: Tuple[int, int], SO_label: int) -> None:
        """ Format: `labels|masked_pos|token_type_lengths|SO_label` """
        line = self.int_array_to_str(labels) + "|" + \
            " ".join(masked_pos) + "|" + \
            self.int_array_to_str(token_type_lengths) + "|" + \
            str(SO_label) + "\n"

        self.writer.write(line)

    def _create_and_save_instances_from_document(self, document: List[List[str]]) -> None:
        """
        :param document: List of tokenized sentence
        """
        current_chunk = []
        current_length = 0
        example_num = 0

        for i, sent in enumerate(document):
            current_chunk.append(sent)
            current_length += len(sent)

            if i == len(document) - 1 or current_length + 2 >= self.config.maxSeqLen:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go
                    # into the `A` (first) sentence.
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    else:
                        current_chunk = []
                        current_length = 0
                        continue

                    tokens_a = [token for chunk in current_chunk[:a_end] for token in chunk]
                    tokens_b = [token for chunk in current_chunk[a_end:] for token in chunk]

                    self.truncate_sent_pair(tokens_a, tokens_b, self.config.maxSeqLen - 2)
                    assert len(tokens_a) >= 1 and len(tokens_b) >= 1

                    SO_label = 1
                    if self.config.NAR and random.random() < 0.5:
                        tokens_a, tokens_b = tokens_b, tokens_a
                        SO_label = 0

                    labels, masked_pos, token_type_lengths = self._create_sample(tokens_a, tokens_b)
                    self._save_sample_to_output_file(labels=labels, masked_pos=masked_pos,
                                                     token_type_lengths=token_type_lengths, SO_label=SO_label)
                    example_num += 1

                current_chunk = []
                current_length = 0


class BartLikePreprocessor(Preprocessor):
    def __init__(self, config_: Config, log_path: str = None):
        super().__init__(config_, log_path)

    def _create_sample(self, tokens: List[str]):
        input_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)

        if self.config.useWWM:
            aligned_tokens = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens)]
            masked_pos = self.masked_pos(aligned_tokens, 0)
        else:
            masked_pos = self.masked_pos(input_ids, 0)

        return input_ids, masked_pos

    def _save_sample_to_output_file(self, labels: List[int], masked_pos: List[str]):
        line = self.int_array_to_str(labels) + "|" + " ".join(masked_pos)
        self.writer.write(line + "\n")

    def _create_and_save_instances_from_document(self, document: List[List[str]]):
        """
        :param document: List of tokenized sentence
        """
        current_chunk = []
        current_length = 0

        for i, sent in enumerate(document):
            current_chunk.append(sent)
            current_length += len(sent)

            if (i == len(document) - 1 or current_length + 2 >= self.config.maxSeqLen) and current_chunk:
                tokens = [token for chunk in current_chunk for token in chunk]
                self.truncate_sent(tokens, self.config.maxSeqLen)

                labels, masked_pos = self._create_sample(tokens)
                self._save_sample_to_output_file(labels, masked_pos)

                current_chunk = [] if current_chunk else current_chunk
                current_length = 0


class T5LikePreprocessor(Preprocessor):
    def __init__(self, config_: Config, log_path: str = None):
        super().__init__(config_, log_path)

    def masked_pos(self, token_ids: Union[List[int], List[List[int]]], bias: int) -> List[str]:
        """ Mask some spans in `token_ids` and return the masked spans ([start, end]) """
        mask_ratio = self.config.maskRatio  # TODO: adjust this
        max_pred_len = self.config.maxPredPerSeq

        # Select masked spans
        idxes = [i for i in range(len(token_ids))]
        random.shuffle(idxes)
        slices = sorted([
            [start, min(len(token_ids), start + random.randint(1, self.config.max_extension_len))]  # [start, end)
            for start in idxes[:int(mask_ratio * len(idxes))]
        ])

        # Merge overlapping spans
        merged_slices = []
        for _slice in slices:
            if len(merged_slices) == 0 or merged_slices[-1][1] < _slice[0]:
                merged_slices.append(_slice)
            else:
                merged_slices[-1][1] = _slice[1]

        slices = merged_slices
        random.shuffle(slices)  # Shuffle slices to prevent only the tailing slices from being lost

        # Convert `span` => `idx span`
        idx_spans = []
        pred_len = 0

        if self.config.useWWM:
            token_ids: List[List[int]]

            accumulate_bias = [bias]
            for i in range(len(token_ids) - 1):
                accumulate_bias.append(accumulate_bias[-1] + len(token_ids[i]))

            for _slice in slices:  # [start, end)
                span_length = sum(len(token_ids[j]) for j in range(*_slice))

                if pred_len + span_length >= max_pred_len:
                    span_length = max_pred_len - pred_len

                _bias = accumulate_bias[_slice[0]]
                if span_length == 1:
                    idx_spans.append(f"{_bias}")
                else:
                    idx_spans.append(f"{_bias}-{_bias + span_length - 1}")  # [left, right]

                pred_len += span_length
                if pred_len >= max_pred_len:
                    break

        else:
            for _slice in slices:
                span_length = _slice[1] - _slice[0]

                if pred_len + span_length >= max_pred_len:
                    span_length -= max_pred_len - pred_len

                if span_length == 1:
                    idx_spans.append(f"{bias + _slice[0]}")
                else:
                    idx_spans.append(f"{bias + _slice[0]}-{bias + span_length - 1}")  # [left, right]

                pred_len += span_length
                if pred_len >= max_pred_len:
                    break

        idx_spans = sorted(idx_spans, key=lambda x: int(x.split("-")[0] if "-" in x else x))

        return idx_spans

    def _create_sample(self, tokens: List[str]):
        input_ids: List[int] = self.tokenizer.convert_tokens_to_ids(tokens)

        if self.config.useWWM:
            aligned_tokens = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens)]
            masked_pos = self.masked_pos(aligned_tokens, 0)
        else:
            masked_pos = self.masked_pos(input_ids, 0)

        return input_ids, masked_pos

    def _save_sample_to_output_file(self, labels: List[int], masked_pos: List[str]):
        if len(masked_pos) > 0:
            line = self.int_array_to_str(labels) + "|" + " ".join(masked_pos)
            self.writer.write(line + "\n")

    def _create_and_save_instances_from_document(self, document: List[List[str]]):
        """
        :param document: List of tokenized sentence
        """
        current_chunk = []
        current_length = 0

        for i, sent in enumerate(document):
            current_chunk.append(sent)
            current_length += len(sent)

            if (i == len(document) - 1 or current_length + 2 >= self.config.maxSeqLen) and current_chunk:
                tokens = [token for chunk in current_chunk for token in chunk]
                self.truncate_sent(tokens, self.config.maxSeqLen)

                labels, masked_pos = self._create_sample(tokens)
                self._save_sample_to_output_file(labels, masked_pos)

                current_chunk = [] if current_chunk else current_chunk
                current_length = 0


def parse_args(args: List[str] = None):
    parser = argparse.ArgumentParser("Preprocessor Argument Parser")

    parser.add_argument("--type", type=str, default="mlm", help="the processor type")
    parser.add_argument("--dupEpochNum", type=int, default=1, help="the epoch number")
    parser.add_argument("--maskRatio", type=float, default=0.15, help="the mask ratio of each sequence")
    parser.add_argument("--maskProb", type=float, default=0.8,
                        help="the probability of replacing a token into [MASK] or its synonym")
    parser.add_argument("--max_extension_len", type=int, default=8, help="Maximum masked items length")
    parser.add_argument("--WWM", action="store_true", help="whether to use the Whole Word Mask mechanism")
    parser.add_argument("--synonym", action="store_true", help="whether to use synonym or [MASK]")

    parser.add_argument("--maxSeqLen", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--maxPredPerSeq", type=int, default=20, help="the maximum tokens number to mask")
    parser.add_argument("--maxSeqLen2", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--maxPredPerSeq2", type=int, default=20, help="the maximum tokens number to mask")

    parser.add_argument("--input_path", "-i", type=str, required=True, help="input path to preprocess")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="output path to save")
    parser.add_argument("--extraSegWordsFile", type=str, default=None,
                        help="Additional dict file for the Chinese tokenizer")
    parser.add_argument("--verboseStep", type=int, default=None,
                        help="the verbose step. None means displaying nothing.")
    parser.add_argument("--NAR", action="store_true", help="Non-autoregressive mode")
    parser.add_argument("--logPath", type=str, default=None, help="path of log")
    parser.add_argument("--processNum", type=int, default=1, help="process number")
    parser.add_argument("--vocabPath", "-v", type=str, required=True, help="vocabulary path")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)

    logPath = args.logPath
    if logPath is None or logPath.lower() == "none":
        logPath = None

    logger = get_logger("main", logPath)

    logger.info(f"preprocessing data from: {args.input_path}")
    logger.info(f"saving to: {args.output_path}")

    config = Config(
        vocabPath=args.vocabPath,
        maskRatio=args.maskRatio,
        max_extension_len=args.max_extension_len,
        useWWM=args.WWM,
        useSynonym=args.synonym,
        maxSeqLen=args.maxSeqLen,
        maxPredPerSeq=args.maxPredPerSeq,
        maxSeqLen2=args.maxSeqLen2,
        maxPredPerSeq2=args.maxPredPerSeq2,
        extraSegWordsFile=args.extraSegWordsFile,
        verboseStep=args.verboseStep,
        NAR=args.NAR,
    )

    if args.type.lower() == "mlm":
        Processor = MLMPreprocessor
    elif args.type.lower() == "bart":
        Processor = BartLikePreprocessor
    else:
        Processor = T5LikePreprocessor

    def wrapper(config_: Config, logpath_: str, input_file_: str, output_file_: str, epoch_: int):
        try:
            processor = Processor(config_, logpath_)
            processor.run(input_file_, output_file_, epoch_)
            logger.info(f"{processor.loggerPrefix} finished successfully!")

        except Exception as e:
            print(e)
            traceback.print_exc()

    # get all input_files; each process deals with one input file.
    input_files = []
    if os.path.isfile(args.input_path):
        input_files.append(args.input_path)
    else:
        for filename in os.listdir(args.input_path):
            if not filename.startswith("."):
                filepath = join(args.input_path, filename)
                if os.path.isfile(filepath):
                    input_files.append(filepath)

    # crate directory to save each epoch data
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for epoch in range(args.dupEpochNum):
        epochPath = join(args.output_path, f"epoch-{epoch}")
        if not os.path.exists(epochPath):
            os.makedirs(epochPath)

    total = args.dupEpochNum * len(input_files)
    poolSize = min(args.processNum, total, multiprocessing.cpu_count())
    logger.info(f"Process number: {poolSize}")

    if poolSize > 1:
        pool = Pool(poolSize)
        for epoch in range(args.dupEpochNum):
            curr_config = copy.deepcopy(config)  # Configure isolation
            if epoch == args.dupEpochNum - 1:
                curr_config.maxSeqLen = curr_config.maxSeqLen2
                curr_config.maxPredPerSeq = curr_config.maxPredPerSeq2

            for index, input_file in enumerate(input_files):
                output_file = join(args.output_path, f"epoch-{epoch}", os.path.basename(input_file))
                logger.info(
                    f"launching worker (epoch {epoch}, file {os.path.basename(input_file)})\n"
                    f"      >>> input file:  {input_file}\n"
                    f"      >>> output file: {output_file}"
                )
                pool.apply_async(wrapper, args=(curr_config, logPath, input_file, output_file, epoch),
                                 error_callback=lambda value: print("[Error]", value))

        pool.close()
        pool.join()

    else:
        for epoch in range(args.dupEpochNum):
            for index, input_file in enumerate(input_files):
                output_file = join(join(args.output_path, f"epoch-{epoch}"), os.path.basename(input_file))
                logger.info(
                    f"launching worker (epoch {epoch}, file {os.path.basename(input_file)})\n"
                    f"      >>> input file: {input_file}\n"
                    f"      >>> output file: {output_file}"
                )
                wrapper(config, logPath, input_files[index], output_file, epoch)

    logger.info("Data preprocessing finished!")
