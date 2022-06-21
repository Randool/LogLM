"""
Author: Da Ma
Date: 2021.12.31
Description: Some preprocessor of transforming normalized text data into tokenized pre-training data.

REFER TO: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L338
"""
import argparse
import collections
import functools
import json
import logging
import os
import random
import traceback
from multiprocessing import Pool
from os.path import join
from typing import List, Tuple, TextIO

import numpy as np
import thulac

from model.tokenizer import LogLMTokenizer
from utils import get_logger

# ========= MLM Instance Object ========= #
MLMInstance = collections.namedtuple("MLMInstance", ["index", "label"])


class MLMConfig:
    def __init__(self, **kwargs) -> None:
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
        self.vocabPath: str = kwargs.pop("vocabPath", None)

        assert self.vocabPath is not None, "vocabPath is allowed None."


class BasePreprocessor:
    inputFile: str
    writer: TextIO
    loggerPrefix: str

    def __init__(self, config: MLMConfig, logPath: str = None):
        self.config = config
        self.logger = get_logger(__name__, logPath)

    def run(self, inputFile: str, outputFile: str, epoch: int):
        self.inputFile = inputFile
        self.writer = open(outputFile, "w")
        self.loggerPrefix = f"epoch {epoch}, file {os.path.basename(inputFile)} ===> "


class MLMPreprocessor(BasePreprocessor):
    def __init__(self, config: MLMConfig, logPath: str = None, debug: bool = False):
        super().__init__(config, logPath)
        self.tokenizer = LogLMTokenizer(vocab_file=self.config.vocabPath)
        self.segmentor = thulac.thulac(user_dict=self.config.extraSegWordsFile, seg_only=True)
        self.debug = debug

    def _padding(self, data: List[int], padding: int) -> List[int]:
        if len(data) < self.config.maxSeqLen:
            data += [padding] * (self.config.maxSeqLen - len(data))
        return data

    def _writeInstanceToOutputFile(
            self,
            tokens: List[str],
            indices: List[int],
            labels: List[str],
            segmentBoundary: int,
            SOLabel: int,
            word_left_bound,
            word_right_bound,
    ) -> None:
        try:
            example = {
                "MaskedToken": self._padding(
                    self.tokenizer.convert_tokens_to_ids(tokens),
                    self.tokenizer.pad_token_id,
                ),
                "segBound": segmentBoundary,
                "SOLabel": SOLabel,
                "MaskedLabel": self._padding(
                    self._gatherLabels(
                        indices, self.tokenizer.convert_tokens_to_ids(labels)
                    ),
                    -100,
                ),  # -100 means ignoring this token
                "wordLeftBound": self._padding(word_left_bound, -1),
                "wordRightBound": self._padding(word_right_bound, -1),
            }
            self.writer.write(f"{json.dumps(example)}\n")
        except ValueError or IndexError:
            errMsg = f"{tokens} can not be masked!"
            if self.logger is not None:
                self.logger.info(errMsg)
            else:
                print(errMsg)

    @staticmethod
    def _truncateSeqPair(tokens_a: List[str], tokens_b: List[str], max_num_tokens: int):
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

    def _tokensToWordTokens(self, tokens: List[str]) -> List[str]:
        text = "".join(self.tokenizer.convert_tokens_to_string(tokens).split())
        if text.startswith("##"):
            text = text[2:]
        segments = self.segmentor.cut(text, text=True)
        return segments.split()

    @staticmethod
    def _wordPieceLength(piece: str) -> int:
        return len(piece) if not piece.startswith("##") else len(piece) - 2

    @staticmethod
    def _isChinesePiece(piece: str) -> bool:
        def _is_chinese_char(cp):
            """Checks whether CP is the codepoint of a CJK character."""
            # This defines a "chinese character" as anything in the CJK Unicode block:
            #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
            #
            # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
            # despite its name. The modern Korean Hangul alphabet is a different block,
            # as is Japanese Hiragana and Katakana. Those alphabets are used to write
            # space-separated words, so they are not treated specially and handled
            # like the all the other languages.
            if (
                    (0x4E00 <= cp <= 0x9FFF)
                    or (0x3400 <= cp <= 0x4DBF)  #
                    or (0x20000 <= cp <= 0x2A6DF)  #
                    or (0x2A700 <= cp <= 0x2B73F)  #
                    or (0x2B740 <= cp <= 0x2B81F)  #
                    or (0x2B820 <= cp <= 0x2CEAF)  #
                    or (0xF900 <= cp <= 0xFAFF)
                    or (0x2F800 <= cp <= 0x2FA1F)  #
            ):  #
                return True

            return False

        return len(piece) == 1 and _is_chinese_char(ord(piece))

    @staticmethod
    def _align(pieces: List[str], words: List[str]) -> List[str]:
        piecesLen = 0
        wordsLen = 0
        wordIndex = 0

        alignedPieces = []
        for piece in pieces:
            # find the smallest wordslen >= pieceslen
            while wordsLen < piecesLen:
                wordsLen += len(words[wordIndex])
                wordIndex += 1

            if MLMPreprocessor._isChinesePiece(piece) and wordsLen != piecesLen:
                alignedPiece = "##" + piece
            else:
                alignedPiece = piece
            alignedPieces.append(alignedPiece)

            piecesLen += MLMPreprocessor._wordPieceLength(piece)

        return alignedPieces

    def _wordPieceFormatChTokens(self, tokens_a: List[str], tokens_b: List[str]) -> List[str]:
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        coarseSegments_a = self._tokensToWordTokens(tokens_a)
        coarseSegments_b = self._tokensToWordTokens(tokens_b)

        coarseSegments = (
                ["[CLS]"] + coarseSegments_a + ["[SEP]"] + coarseSegments_b + ["[SEP]"]
        )

        segments = []
        for coarseSegment in coarseSegments:
            segments += coarseSegment.split()

        if self.debug:
            segmentsText = " ".join(segments)
            piecesText = self.tokenizer.convert_tokens_to_string(tokens)
            print(f"words: {segmentsText}")
            print(f"pieces: {piecesText}")

            piecesJoinText = " ".join(tokens)
            print(f"pieces splits: {piecesJoinText}")

        return self._align(tokens, segments)

    def _createMLMPredictions(self, tokens_a: List[str], tokens_b: List[str]) -> Tuple[List[str], List[int], List[str]]:
        """Creates the predictions for the masked LM objective."""
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        cand_indexes = []
        word_left_bound = []
        word_right_bound = []

        if not self.config.useWWM:
            for (i, token) in enumerate(tokens):
                if token == "[CLS]" or token == "[SEP]":
                    continue
                cand_indexes.append([i])
        else:
            # Whole Word Masking means that if we mask all the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmax-ed
            # over the entire vocabulary.

            # NOTE: For chinese, "Whole Word" indicates a phrase
            chWordPieceTokens = self._wordPieceFormatChTokens(tokens_a, tokens_b)

            lengths = []
            pre_len = 0
            for (i, token) in enumerate(chWordPieceTokens):
                if token == "[CLS]" or token == "[SEP]":
                    if pre_len:
                        lengths.append(pre_len)
                    pre_len = 1
                    continue

                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                    pre_len += 1
                else:
                    if pre_len:
                        lengths.append(pre_len)
                    pre_len = 1
                    cand_indexes.append([i])

            if pre_len:
                lengths.append(pre_len)

            # build boundaries
            cur = 0
            for length in lengths:
                word_left_bound += [cur] * length
                cur += length
                word_right_bound += [cur] * length

        random.shuffle(cand_indexes)

        # copy tokens
        output_tokens = list(tokens)

        num_to_predict = min(
            self.config.maxPredPerSeq,
            max(1, int(round(len(tokens) * self.config.maskRatio))),
        )

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    # FIXME: replace to synonym
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        # NOTE: different random range for different tokenizers
                        start = len(self.tokenizer.special_tokens_map)
                        masked_token = self.tokenizer.convert_ids_to_tokens(
                            random.randint(start, len(self.tokenizer) - 1)
                        )

                output_tokens[index] = masked_token
                masked_lms.append(MLMInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        return (
            output_tokens,
            list(map(lambda x: x.index, masked_lms)),
            list(map(lambda x: x.label, masked_lms)),
            word_left_bound,
            word_right_bound,
        )

    @staticmethod
    def _gatherLabels(indices: List[int], labels: List[int]) -> List[int]:
        gatheredLabels = np.full((max(indices) + 1,), -100)
        gatheredLabels[indices] = labels
        return gatheredLabels.tolist()

    def _createAndSaveInstancesFromDocument(self, document: List[List[str]]) -> None:
        """Crate instances from the given document and save to the output file."""
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = self.config.maxSeqLen - 3

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyway, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if random.random() < self.config.shortSeqProb:
            target_seq_length = random.randint(2, max_num_tokens)

        # We DON'T just concatenate all the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input. Following MacBERT, we adopt the SOP(Sentence Order Prediction) task.
        current_chunk = []
        current_length = 0
        i = 0

        exampleNum = 0

        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    else:
                        current_chunk = []
                        current_length = 0
                        i += 1
                        continue

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    self._truncateSeqPair(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    # swap, SOP
                    SOLabel = 1
                    if random.random() < 0.5:
                        tokens_a, tokens_b = tokens_b, tokens_a
                        SOLabel = 0

                    segmentBoundary = 2 + len(tokens_a)  # [CLS] A [SEP]

                    tokens, indices, labels, word_left_bound, word_right_bound = self._createMLMPredictions(tokens_a,
                                                                                                            tokens_b)

                    if self.debug:
                        maskedPiecesSplit = " ".join(tokens)
                        maskedIndex = " ".join(map(str, indices))
                        maskedLabel = " ".join(labels)
                        print(f"masked pieces split: {maskedPiecesSplit}")
                        print(f"masked pieces index: {maskedIndex}")
                        print(f"masked pieces label: {maskedLabel}")
                        print(f"SOP label: {SOLabel}")
                        print(f"word left bound: {word_left_bound}")
                        print(f"word right bound: {word_right_bound}")
                        print()

                    self._writeInstanceToOutputFile(tokens, indices, labels, segmentBoundary, SOLabel,
                                                    word_left_bound, word_right_bound, )
                    exampleNum += 1

                current_chunk = []
                current_length = 0

            i += 1

        if not exampleNum and self.debug:
            print(f"For this doc, the target sequence length is {target_seq_length}, "
                  f"which is too short to merge two sentences into one sequence.")

    @staticmethod
    def _cleanText(text: str) -> str:
        text = text.lower()  # lowercase
        text = text.replace("#", " ")  # remote all the '#' character
        return text.strip()

    def _createTrainingInstancesForSingleFile(self) -> None:
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
                line = self._cleanText(line)

                # Empty lines are used as document delimiters
                if not line:
                    if not document:
                        break

                    if self.debug:
                        print(f"=========== TEST INFO OF DOC {docNum} ===========\n")

                    # create instances for the document
                    self._createAndSaveInstancesFromDocument(document)
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
        super().run(inputFile, outputFile, epoch)
        self._createTrainingInstancesForSingleFile()
        self.writer.close()


logging.basicConfig(level=logging.INFO)


def parse():
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

    return parser.parse_args()


def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            print("Exception in " + func.__name__)
            traceback.print_exc()

    return wrapped_func


def main():
    args = parse()

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
            maskProb=args.maskProb,
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
    poolSize = min(args.processNum, total)
    logger.info(f"Process number: {poolSize}")

    if poolSize > 1:
        pool = Pool(poolSize)
        for epoch in range(args.dupEpochNum):
            for index, inputFile in enumerate(inputFiles):
                outputFile = join(join(args.outputPath, f"epoch-{epoch}"), os.path.basename(inputFile))
                logger.info(
                    f"launching worker (epoch {epoch}, file {os.path.basename(inputFile)})\n"
                    f"      >>> input file: {inputFile}\n"
                    f"      >>> output file: {outputFile}"
                )
                pool.apply_async(wrapper, args=(config, logPath, inputFile, outputFile, epoch))

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

    logger.info("preprocess the data successfully!")


if __name__ == '__main__':
    main()
