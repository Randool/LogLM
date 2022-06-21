"""
@File        :  build_pretrain_data.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/18
@Version     :  1.0
@Description :  None
"""
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpora", type=str, nargs="+", required=True, help="Corpora for pretrain")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length")
    return parser.parse_args()


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
