"""
Author: Da Ma
Date: 2021.12.31
Description: Some preprocessor of transforming normalized text data into tokenized pre-training data.

REFER TO: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L338
"""
import argparse
import copy
import functools
import multiprocessing
import os
import random
import traceback
from math import log2, ceil
from multiprocessing import Pool
from os.path import join
from typing import List, Tuple, TextIO, Union, Iterable

import thulac

from model import LogLMTokenizer
from utils.logger import get_logger


class MLMConfig:
    def __init__(self, vocabPath: str, **kwargs) -> None:
        """Define several setting parameters."""
        self.maskRatio = kwargs.pop("maskRatio", 0.15)
        self.maskProb = kwargs.pop("maskProb", 0.8)
        self.useWWM = kwargs.pop("useWWM", False)
        self.useSynonym = kwargs.pop("useSynonym", False)
        self.maxSeqLen = kwargs.pop("maxSeqLen", 512)
        self.shortSeqProb = kwargs.pop("shortSeqProb", 0.1)
        self.maxPredPerSeq = kwargs.pop("maxPredPerSeq", 20)
        self.extraSegWordsFile: str = kwargs.pop("extraSegWordsFile", None)
        self.verboseStep: int = kwargs.pop("verboseStep", None)
        self.vocabPath = vocabPath


class MLMPreprocessor:
    inputFile: str
    writer: TextIO
    loggerPrefix: str

    def __init__(self, config: MLMConfig, logPath: str = None, debug: bool = False):
        self.config = config
        self.logger = get_logger(__name__, logPath)
        self.tokenizer = LogLMTokenizer(vocab_file=self.config.vocabPath)
        self.segmentor = thulac.thulac(user_dict=self.config.extraSegWordsFile, seg_only=True)
        self.debug = debug

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
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    @staticmethod
    def prompt_length(length: int) -> int:
        return ceil(log2(max(1, length)))

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

    def _masked_pos(self, token_ids: Union[List[int], List[List[int]]], bias: int) -> List[int]:
        """ 只需保存被mask的token的位置即可。另外，使用“水库抽样”算法限制最大预测个数 """
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

        return sorted(pos)

    def _create_sample(self, tokens_a: List[str], tokens_b: List[str]):
        prompt_len = self.prompt_length(len(tokens_a) + len(tokens_b))
        token_type_lengths = (prompt_len + len(tokens_a) + 1, len(tokens_b) + 1)

        prompt_token_ids = self.tokenizer.prompt_token_ids[:prompt_len]
        sep_token_id = [self.tokenizer.sep_token_id]

        input_ids_a: List[int] = self.tokenizer.convert_tokens_to_ids(tokens_a)
        input_ids_b: List[int] = self.tokenizer.convert_tokens_to_ids(tokens_b)
        labels = prompt_token_ids + input_ids_a + sep_token_id + input_ids_b + sep_token_id

        if self.config.useWWM:
            aligned_tokens_a = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens_a)]
            aligned_tokens_b = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in self.do_align(tokens_b)]
            masked_pos = self._masked_pos(aligned_tokens_a, prompt_len) + \
                         self._masked_pos(aligned_tokens_b, prompt_len + len(input_ids_a) + 1)
        else:
            masked_pos = self._masked_pos(input_ids_a, prompt_len) + \
                         self._masked_pos(input_ids_b, prompt_len + len(input_ids_a) + 1)

        return labels, masked_pos, token_type_lengths, prompt_len

    def _save_sample_to_output_file(self, labels: List[int], masked_pos: List[int],
                                    token_type_lengths: Tuple[int, int], prompt_length: int, SO_label: int):
        """ Format: `labels|masked_pos|token_type_lengths|prompt_length|SO_label` """

        def int_array_to_str(array: Iterable[int]) -> str:
            return " ".join(map(str, array))

        line = int_array_to_str(labels) + "|" + \
               int_array_to_str(masked_pos) + "|" + \
               int_array_to_str(token_type_lengths) + "|" + \
               str(prompt_length) + "|" + str(SO_label) + "\n"
        self.writer.write(line)

    def _create_and_save_instances_from_document(self, document: List[List[str]]):
        current_chunk = []
        current_length = 0
        example_num = 0

        for i, sent in enumerate(document):
            current_chunk.append(sent)
            current_length += len(sent)
            prompt_len = self.prompt_length(current_length)

            if i == len(document) - 1 or prompt_len + current_length + 2 >= self.config.maxSeqLen:
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

                    self.truncate_sent_pair(tokens_a, tokens_b, self.config.maxSeqLen - 2 - prompt_len)
                    assert len(tokens_a) >= 1 and len(tokens_b) >= 1

                    SO_label = 1
                    if random.random() < 0.5:
                        tokens_a, tokens_b = tokens_b, tokens_a
                        SO_label = 0

                    labels, masked_pos, token_type_lengths, prompt_len = self._create_sample(tokens_a, tokens_b)
                    self._save_sample_to_output_file(labels=labels, masked_pos=masked_pos,
                                                     token_type_lengths=token_type_lengths,
                                                     prompt_length=prompt_len, SO_label=SO_label)
                    example_num += 1

                current_chunk = []
                current_length = 0

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

        with open(self.inputFile, "r") as reader:
            while True:
                line = reader.readline().strip()
                line = self.clean_text(line)

                # Empty lines are used as document delimiters
                if not line:
                    if not document:
                        break

                    if self.debug:
                        print(f"=========== TEST INFO OF DOC {docNum} ===========\n")

                    # create instances for the document
                    self._create_and_save_instances_from_document(document)
                    docNum += 1

                    if self.debug:
                        print("************************************************\n\n")

                    if (
                            self.config.verboseStep is not None
                            and not (docNum % self.config.verboseStep)
                            and self.logger is not None
                    ):
                        self.logger.info(
                            f"{self.loggerPrefix}finish processing {docNum} documents!"
                        )

                    document = []

                tokens = self.tokenizer.tokenize(line)
                if tokens:
                    document.append(tokens)

    def run(self, inputFile: str, outputFile: str, epoch: int):
        self.inputFile = inputFile
        self.writer = open(outputFile, "w")
        self.loggerPrefix = f"epoch {epoch}, file {os.path.basename(inputFile)} ===> "
        self._create_training_instances_for_single_file()
        self.writer.close()


def parse_args(args: List[str] = None):
    parser = argparse.ArgumentParser("Preprocessor Argument Parser")

    parser.add_argument("--type", type=str, default="mlm", help="the processor type")
    parser.add_argument("--dupEpochNum", type=int, default=1, help="the epoch number")
    parser.add_argument("--maskRatio", type=float, default=0.15, help="the mask ratio of each sequence")
    parser.add_argument("--maskProb", type=float, default=0.8,
                        help="the probability of replacing a token into [MASK] or its synonym")
    parser.add_argument("--WWM", action="store_true", help="whether to use the Whole Word Mask mechanism")
    parser.add_argument("--synonym", action="store_true", help="whether to use synonym or [MASK]")
    parser.add_argument("--maxSeqLen", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--inputPath", "-i", type=str, required=True, help="input path to preprocess")
    parser.add_argument("--outputPath", "-o", type=str, required=True, help="output path to save")
    parser.add_argument("--shortSeqProb", type=float, default=0.1, help="the probability of using short sequence")
    parser.add_argument("--maxPredPerSeq", type=int, default=20, help="the maximum tokens number to mask")
    parser.add_argument("--extraSegWordsFile", type=str, default=None,
                        help="Additional dict file for the Chinese tokenizer")
    parser.add_argument("--verboseStep", type=int, default=None,
                        help="the verbose step. None means displaying nothing.")
    parser.add_argument("--logPath", type=str, default=None, help="path of log")
    parser.add_argument("--processNum", type=int, default=1, help="process number")
    parser.add_argument("--vocabPath", "-v", type=str, required=True, help="vocabulary path")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    return parser.parse_args(args)


def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            print("Exception in " + func.__name__)
            traceback.print_exc()

    return wrapped_func


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)

    logPath = args.logPath
    if logPath is None or logPath.lower() == "none":
        logPath = None

    logger = get_logger("main", logPath)

    logger.info(f"preprocessing data from: {args.inputPath}")
    logger.info(f"saving to: {args.outputPath}")

    if args.type.lower() == "mlm":
        config = MLMConfig(
            maskRatio=args.maskRatio,
            useWWM=args.WWM,
            useSynonym=args.synonym,
            maxSeqLen=args.maxSeqLen,
            shortSeqProb=args.shortSeqProb,
            maxPredPerSeq=args.maxPredPerSeq,
            extraSegWordsFile=args.extraSegWordsFile,
            verboseStep=args.verboseStep,
            vocabPath=args.vocabPath,
            processNum=args.processNum,
        )
        Processor = MLMPreprocessor
    else:
        raise NotImplementedError()


    @trace_unhandled_exceptions
    def wrapper(config_: MLMConfig, logpath_: str, inputFile_: str, outputFile_: str, epoch_: int):
        try:
            processor = Processor(config_, logpath_)
            processor.run(inputFile_, outputFile_, epoch_)

            logger.info(f"{processor.loggerPrefix} finished successfully!")

        except Exception as e:
            logger.info(f"Process: epoch {epoch_}, file {os.path.basename(inputFile_)}")
            raise Exception(e)


    # get all inputFiles; each process deals with one input file.
    inputFiles = []
    if os.path.isfile(args.inputPath):
        inputFiles.append(args.inputPath)
    else:
        for filename in os.listdir(args.inputPath):
            if not filename.startswith("."):
                filepath = join(args.inputPath, filename)
                if os.path.isfile(filepath):
                    inputFiles.append(filepath)

    # crate directory to save each epoch data
    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)

    for epoch in range(args.dupEpochNum):
        epochPath = join(args.outputPath, f"epoch-{epoch}")
        if not os.path.exists(epochPath):
            os.makedirs(epochPath)

    total = args.dupEpochNum * len(inputFiles)
    poolSize = min(args.processNum, total, multiprocessing.cpu_count())
    logger.info(f"Process number: {poolSize}")

    if poolSize > 1:
        pool = Pool(poolSize)
        for epoch in range(args.dupEpochNum):
            curr_config = copy.deepcopy(config)  # Configure isolation
            if epoch == args.dupEpochNum - 1:
                curr_config.maxSeqLen = 512

            for index, inputFile in enumerate(inputFiles):
                outputFile = join(args.outputPath, f"epoch-{epoch}", os.path.basename(inputFile))
                logger.info(
                    f"launching worker (epoch {epoch}, file {os.path.basename(inputFile)})\n"
                    f"      >>> input file: {inputFile}\n"
                    f"      >>> output file: {outputFile}"
                )
                pool.apply_async(wrapper, args=(curr_config, logPath, inputFile, outputFile, epoch))

        pool.close()
        pool.join()

    else:
        for epoch in range(args.dupEpochNum):
            for index, inputFile in enumerate(inputFiles):
                outputFile = join(join(args.outputPath, f"epoch-{epoch}"), os.path.basename(inputFile))
                logger.info(
                    f"launching worker (epoch {epoch}, file {os.path.basename(inputFile)})\n"
                    f"      >>> input file: {inputFile}\n"
                    f"      >>> output file: {outputFile}"
                )
                wrapper(config, logPath, inputFiles[index], outputFile, epoch)

    logger.info("Data preprocessing finished!")
