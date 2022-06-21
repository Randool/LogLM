"""
Author: Da Ma
Date: 2022.01.05
Description: Chinese Bert Tokenizer Trainer
"""
import argparse
import os

from tokenizers.implementations import BertWordPieceTokenizer


def parse():
    parser = argparse.ArgumentParser("Chinese Bert Tokenizer Trainer")
    parser.add_argument("--inputFileList", type=str, nargs="+", help="input file list")
    parser.add_argument("--savePath", type=str, help="path to save the vocabulary")
    parser.add_argument("--vocabSize", type=int, default=20000, help="vocabulary size")
    parser.add_argument("--minFreq", type=int, default=2, help="minimum frequency")
    parser.add_argument("--limitAlphabet", type=int, default=10000, help="limit alphabet")
    parser.add_argument("--specialTokens", type=str, nargs="+", help="special tokens")
    return parser.parse_args()


def train(args):
    tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True)

    tokenizer.train(
        args.inputFileList,
        vocab_size=args.vocabSize,
        min_frequency=args.minFreq,
        show_progress=True,
        special_tokens=args.specialTokens,
        limit_alphabet=args.limitAlphabet,
        wordpieces_prefix="##",
    )

    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)
    tokenizer.save_model(args.savePath)


if __name__ == "__main__":
    train(parse())
