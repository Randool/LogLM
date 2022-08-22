"""
@File        :  patch_log_base_to_ckpts.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/8/21
@Version     :  1.0
@Description :  之前的代码训练的结果在保存tokenizer配置时会忽略log_base这个关键参数
"""
import argparse
import json
import re
from glob import glob
from os.path import realpath, join, expanduser


def parse_args():
    CKPT_ZOO = join(expanduser("~"), "Dataset/NLP/TencentMedicalPretrain/corpus/ckpts")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default=CKPT_ZOO, help="Checkpoint zoo")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\nCheckpoint zoo: {args.dir}\n")

    tokenizer_cfgs = sorted(glob(join(realpath(args.dir), "*", "checkpoint-*", "tokenizer_config.json")))

    for cfg_file in tokenizer_cfgs:
        log_base = re.search(r"log\d+(.\d+)?", cfg_file).group()
        is_float = "." in log_base
        log_base = float(log_base[3:]) if is_float else int(log_base[3:])

        with open(cfg_file, encoding="utf8") as f:
            config: dict = json.load(f)

        if config.get("log_base", -1000) == log_base:
            print(f"[Pass] {cfg_file}")
            continue
        else:
            print(f"[{log_base}] {cfg_file}")

        config["log_base"] = log_base
        with open(cfg_file, "w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
