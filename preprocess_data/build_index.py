"""
Author: Da Ma
Date: 2022.01.07
"""
import argparse
import os
from multiprocessing import Pool


class Indexer:
    def __init__(self, inputFile: str, outputFile: str) -> None:
        self.inputFile = inputFile
        self.outputFile = outputFile

    def run(self):
        with open(self.inputFile, "rb") as reader:
            with open(self.outputFile, "w") as writer:
                while True:
                    offset = reader.tell()
                    line = reader.readline()
                    if not line:
                        break
                    writer.write(f"{offset}\n")


def parse():
    parser = argparse.ArgumentParser("Index Building Launcher")

    parser.add_argument("--inputPath", type=str, help="input path")
    parser.add_argument("--processNum", type=int, default=1, help="process number")

    return parser.parse_args()


def main():
    args = parse()

    assert os.path.exists(args.inputPath), f"{args.inputPath} does not exist!"

    inputFiles = []
    for epochFile in os.listdir(args.inputPath):
        epochPath = os.path.join(args.inputPath, epochFile)
        if os.path.isfile(epochPath) and not epochPath.startswith(".") and not epochPath.endswith("index"):
            inputFiles.append(epochPath)

    pool = Pool(args.processNum)

    for inputFile in inputFiles:
        outputFile = os.path.join(args.inputPath, os.path.basename(inputFile) + ".index")
        indexer = Indexer(inputFile, outputFile)
        pool.apply_async(indexer.run, args=())

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
