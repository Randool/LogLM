"""
Author: Da Ma
Date: 2021.12.31
Description: the main program of normalizing data.
"""
import argparse
import os
import random
from os.path import join, expanduser
from typing import List

from tqdm import tqdm

from utils.logger import get_logger
from utils.utils import cutSentByNumOrder, cutTitle, sentSplit, normalizeSent, removeDupTitle

logger = get_logger("normalizer")


def parse():
    parser = argparse.ArgumentParser("Preprocessing Normalizer Argument Parser")

    _HOME = expanduser("~")
    _corpus_dir = os.path.join(_HOME, "Dataset/NLP/TencentMedicalPretrain/corpus")
    _raw_corpus_dir = join(_corpus_dir, "15.medical_plm_corpus")
    _normalized_dir = join(_corpus_dir, "normalized")
    _dev_normalized_dir = join(_corpus_dir, "normalized_dev")

    parser.add_argument("--type", type=str, default="mlm", help="the normalizer type")
    parser.add_argument("--input_path", "-i", type=str, default=_raw_corpus_dir, help="input data path")
    parser.add_argument("--output_path", "-o", type=str, default=_normalized_dir, help="output path for train")
    parser.add_argument("--dev_output_path", type=str, default=_dev_normalized_dir, help="output path for dev")
    parser.add_argument("--exclude_files", "-x", type=str, nargs="*", default=["drug_clean.txt"], help="File excluded")
    parser.add_argument("--rate", type=float, default=1.0, help="the data rate to select")
    parser.add_argument("--num_doc_pre_file", "-n", type=int, default=100000,
                        help="the document number per file. -1 means saving all documents into one single file.")
    parser.add_argument("--num_dev_doc", type=int, default=8000, help="Number of doc for dev")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    return parser.parse_args()


class MLMNormalizer:
    def __init__(self, input_path: str, output_path: str, dev_output_path: str, exclude_files: List[str],
                 rate: float, num_dev_doc: int, num_doc_pre_file: int = -1, seed=42):
        """This normalizer is for MLM task.
        NOTE: This normalizer supports to select a subset of total examples.

        Args:
            rate (float): the rate of examples to select.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.dev_output_path = dev_output_path
        self.exclude_files = exclude_files
        self.num_dev_doc = num_dev_doc
        self.num_doc_pre_file = num_doc_pre_file

        self.rate = rate

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(dev_output_path, exist_ok=True)

        self.doc_file_id = 0
        self.outf = open(os.path.join(output_path, f"doc.{self.doc_file_id}"), "w")

        self.dev_buff = []
        random.seed(seed)

    def __del__(self):
        if not self.outf.closed:
            self.outf.close()

    @staticmethod
    def _count_lines(input_file: str) -> int:
        cnt = 0
        with open(input_file, "r") as inf:
            for line in inf.readlines():
                if line.strip():
                    cnt += 1
        return cnt

    def _run_helper(self, input_file: str, output_path: str):
        """Normalize for the specific file."""
        total = self._count_lines(input_file)
        logger.info(f"total: {total} | selected: {int(total * self.rate)}")
        doc_cnt = 0
        train_doc_cnt = 0

        with open(input_file, "r") as inf:
            for line in tqdm(inf.readlines(), ncols=64):
                line = line.strip()
                if line:
                    """
                    1. Cut the sentence by punctuation first.
                    2. Cut the sentences more precisely. Consider the
                       number order, e.g. 1. (1) and so on.
                    3. Clean each sentence:
                        (1) remove leading punctuations
                        (2) remove trivial sentences, e.g. 1。
                        (3) split [\\t] delimited sentences
                        (4) remove spaces
                    """
                    coarsedSents = sentSplit(line)

                    # cut sentence further according to the number order, e.g. (1) 1、
                    finedSents = []
                    for coarsedSent in coarsedSents:
                        finedSents += cutSentByNumOrder(coarsedSent)

                    # select the title specifically
                    finedSents = cutTitle(finedSents[0]) + finedSents[1:]

                    # normalize
                    sents = []
                    for finedSent in finedSents:
                        sents += normalizeSent(finedSent)

                    # remove duplicated title
                    sents = removeDupTitle(sents)

                    # write sents to the `train_dir` or `dev_dir`
                    chunk = "\n".join(map(lambda x: x.strip(), sents)) + "\n\n"

                    if len(self.dev_buff) < self.num_dev_doc:
                        self.dev_buff.append(chunk)
                    else:
                        # Using reservoir sampling algorithm to decide whether the
                        # chunk be written to train_dev or dev_buff
                        dice = random.randint(0, doc_cnt)
                        if dice < self.num_dev_doc:
                            chunk, self.dev_buff[dice] = self.dev_buff[dice], chunk

                        train_doc_cnt += 1
                        self.outf.write(chunk)

                        if (self.num_doc_pre_file > -1) and not (train_doc_cnt % self.num_doc_pre_file):
                            self.outf.close()
                            self.doc_file_id += 1
                            self.outf = open(os.path.join(output_path, f"doc.{self.doc_file_id}"), "w")

                    doc_cnt += 1

    def run(self):
        # collect all the input files
        if os.path.isdir(self.input_path):
            input_files = [
                os.path.join(self.input_path, file)
                for file in os.listdir(self.input_path)
                if (
                        not file.startswith(".")
                        and os.path.isfile(os.path.join(self.input_path, file))
                        and file not in self.exclude_files
                )
            ]
        else:
            input_files = [self.input_path]
        input_files = sorted(input_files)

        # normalize for each file, respectively
        os.makedirs(self.output_path, exist_ok=True)

        for input_file in input_files:
            logger.info(f"normalize file: {input_file}")
            self._run_helper(input_file, self.output_path)

        self.outf.close()

        # Write selected doc to dev_dir
        with open(os.path.join(self.dev_output_path, "doc.0"), "w") as f:
            for chunk in self.dev_buff:
                f.write(chunk)


if __name__ == "__main__":
    args = parse()
    random.seed(args.seed)
    logger = get_logger("main")

    logger.info(f"normalizing data from: {args.input_path} with a rate of {args.rate}")
    logger.info(f"saving to: {args.output_path}")

    if args.type.lower() == "mlm":
        normalizer = MLMNormalizer(
            input_path=args.input_path, output_path=args.output_path, dev_output_path=args.dev_output_path,
            exclude_files=args.exclude_files, rate=args.rate, num_dev_doc=args.num_dev_doc,
            num_doc_pre_file=args.num_doc_pre_file, seed=args.seed
        )
    else:
        raise NotImplementedError()

    normalizer.run()

    logger.info("normalize the data successfully!")
