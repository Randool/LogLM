"""
@File        :  mlm_acc_with_frequency.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/9
@Version     :  1.0
@Description :  测试MLM预测准确率和词频的关系
"""
from collections import defaultdict, Counter
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DatasetForShuffledFile
from model import LogLMTokenizer, LogLMForPreTraining


def dict_to_device(inputs: dict, device):
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs


def pred_acc_on_different_freq():
    """ 测试模型在不同词频下的预测准确率. 此时不做任何MASK, 且测试模型均带有decoder """
    device = "cuda:0"

    corpus_path = join(expanduser("~"), "Dataset/NLP/TencentMedicalPretrain/corpus")
    model_cfg = "base_ndec2_log1024"
    model_path = join(corpus_path, f"ckpts/{model_cfg}/checkpoint-950000")
    corpus = join(corpus_path, "processed/dev/dev.data")

    tokenizer = LogLMTokenizer.from_pretrained(model_path)
    model = LogLMForPreTraining.from_pretrained(model_path).to(device).eval()
    assert model.non_casual_decoder is not None
    model.casual_decoder = None

    pred_stat = defaultdict(lambda: [0, 0])  # [correct, total]
    error_token_ids_freq, token_ids_freq = Counter(), Counter()
    correct, total = 0, 0
    n_samples = 5000
    devset = DatasetForShuffledFile(corpus, mask_token_id=tokenizer.mask_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    log_base=float(model_cfg.split("_")[-1][3:]))
    dataloader = DataLoader(devset, batch_size=1, collate_fn=devset.collate_fn)

    with torch.no_grad(), tqdm(enumerate(dataloader, start=1), total=n_samples, ncols=60) as pbar:
        for i, inputs in pbar:
            if i >= n_samples:
                break
            labels = inputs["input_ids"] = inputs["labels"]
            inputs = dict_to_device(inputs, device)
            ndec_mlm_pred = model.forward(**inputs).ndec_prediction_logits.argmax(dim=-1).cpu()[0]
            labels = labels[0]
            pred_result = labels.eq(ndec_mlm_pred).tolist()

            for res, id_ in zip(pred_result[1:-1], labels.tolist()[1:-1]):
                pred_stat[id_][0] += res
                pred_stat[id_][1] += 1
                correct += res
                total += 1

                token_ids_freq[id_] += 1
                if not res:
                    error_token_ids_freq[id_] += 1

            pbar.set_postfix({"acc": f"{(correct / total):.2%}"})

    pred_stat = dict(pred_stat)
    with open("./exp/error_token_ids.txt", "w", encoding="utf8") as f:
        error_token_ids_ = sorted(error_token_ids_freq.items(), key=lambda x: x[1], reverse=True)
        for error_token_id, times in error_token_ids_:
            total_times = token_ids_freq[error_token_id]
            f.write(f"{tokenizer.convert_ids_to_tokens(error_token_id)}\t{times}\t{total_times}\n")

    df = pd.DataFrame(pred_stat).T
    df = df.rename(columns={0: 'correct', 1: 'total'})
    df["acc_rate"] = df["correct"] / df["total"]
    df["log10_total"] = np.log10(df['total'])

    bins = np.linspace(0.9, round(df['log10_total'].max()), 13)
    df["log10_freq"] = pd.cut(df["log10_total"], bins=bins)
    sns.boxplot(x='log10_freq', y='acc_rate', data=df)
    plt.xticks(rotation=45)
    plt.title(f"{model_cfg} | acc rate: {(correct / total):.2%}")
    plt.tight_layout()
    plt.savefig(f'figs/boxplot_{model_cfg}.png', dpi=200)


def main():
    pred_acc_on_different_freq()


if __name__ == '__main__':
    main()
