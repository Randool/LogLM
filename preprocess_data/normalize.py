"""
Author: Da Ma
Date: 2021.12.31
Description: the main program of normalizing data.
"""
import argparse
import os
import random

from tqdm import tqdm

from utils import cutSentByNumOrder, cutTitle, get_logger, sentSplit, normalizeSent, removeDupTitle

logger = get_logger("normalizer")


class BaseNormalizer:
    def __init__(self, inputPath: str, outputPath: str, docNumPerFile: int = -1):
        """
        Args:
            inputPath (str): input directory or input file path
            outputPath (str): output file path
        """
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.docNumPerFile = docNumPerFile

    def run(self):
        raise NotImplementedError()


class MLMNormalizer(BaseNormalizer):
    def __init__(self, inputPath: str, outputPath: str, rate: float, docNumPerFile: int = -1):
        """This normalizer is for MLM task.
        NOTE: This normalizer supports to select a subset of total examples.

        Args:
            rate (float): the rate of examples to select.
        """
        super().__init__(inputPath, outputPath, docNumPerFile)

        self.rate = rate

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        outputFile = (
            self.outputPath
            if self.docNumPerFile == -1
            else os.path.join(self.outputPath, "doc.0")
        )
        self.docNum = 0
        self.outf = open(outputFile, "w")

    @staticmethod
    def _countLines(inputFile: str) -> int:
        cnt = 0
        with open(inputFile, "r") as inf:
            for line in inf.readlines():
                if line.strip():
                    cnt += 1
        return cnt

    def _runHelper(self, inputFile: str, outputPath: str):
        """Normalize for the specific file."""
        total = self._countLines(inputFile)
        selected = random.sample(list(range(total)), int(total * self.rate))
        selected.sort()

        logger.info(f"total: {total} | selected: {int(total * self.rate)}")

        lineIndex = 0
        selectedExampleIndex = 0
        with open(inputFile, "r") as inf:
            for line in tqdm(inf.readlines(), ncols=64):
                line = line.strip()
                if selectedExampleIndex >= len(selected):
                    break

                if line:
                    if lineIndex == selected[selectedExampleIndex]:
                        # tokenize the document into sentence by sentence.
                        # =============== NOTE ============== #
                        # FORMAT: each doc is seperated by an empty line.
                        #
                        # SOME NORMALIZATION PROCEDURE: [FIXME]
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

                        # write sents to the output file
                        self.outf.write(
                            "\n".join(map(lambda x: x.strip(), sents)) + "\n\n"
                        )

                        selectedExampleIndex += 1

                        if self.docNumPerFile > -1 and not (
                                selectedExampleIndex % self.docNumPerFile
                        ):
                            self.outf.close()
                            self.docNum += 1
                            self.outf = open(
                                os.path.join(outputPath, f"doc.{self.docNum}"), "w"
                            )
                    lineIndex += 1

    def run(self):
        # collect all the input files
        if os.path.isdir(self.inputPath):
            inputFiles = [
                os.path.join(self.inputPath, file)
                for file in os.listdir(self.inputPath)
                if not file.startswith(".") and os.path.isfile(os.path.join(self.inputPath, file))
            ]
        else:
            inputFiles = [self.inputPath]
        inputFiles = sorted(inputFiles)

        # normalize for each file, respectively
        if self.docNumPerFile == -1:
            with open(self.outputPath, "w"):
                pass
        elif not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        for inputFile in inputFiles:
            logger.info(f"normalize file: {inputFile}")
            self._runHelper(inputFile, self.outputPath)

        self.outf.close()


def parse():
    parser = argparse.ArgumentParser("Preprocessing Normalizer Argument Parser")

    parser.add_argument("--type", type=str, default="mlm", help="the normalizer type")
    parser.add_argument("--inputPath", "-i", type=str, required=True, help="input data path")
    parser.add_argument("--outputPath", "-o", type=str, required=True, help="output data path")
    parser.add_argument("--rate", type=float, default=1.0, help="the data rate to select")
    parser.add_argument("--docNumPerFile", type=int, default=100000,
                        help="the document number per file. -1 means saving all documents into one single file.")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    random.seed(args.seed)

    logger = get_logger("main")

    logger.info(f"normalizing data from: {args.inputPath} with a rate of {args.rate}")
    logger.info(f"saving to: {args.outputPath}")

    if args.type.lower() == "mlm":
        normalizer = MLMNormalizer(args.inputPath, args.outputPath, args.rate, args.docNumPerFile)
    else:
        raise NotImplementedError()

    normalizer.run()

    logger.info("normalize the data successfully!")
