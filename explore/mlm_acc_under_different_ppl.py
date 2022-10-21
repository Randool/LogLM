"""
@File        :  mlm_acc_under_different_ppl.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/10/13
@Version     :  1.0
@Description :  计算不同PPL的句子是否会影响模型预测的准确率
"""
from os.path import join, expanduser

import kenlm
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from model import VCLSTokenizer, LogLMForPreTraining

device = "cuda" if torch.cuda.is_available() else "cpu"

corpus_path = join(expanduser("~"), "Dataset/NLP/TencentMedicalPretrain/corpus")
test_file = join(corpus_path, "15.medical_plm_corpus/mfk.txt")
# ckpt_path = join(corpus_path, "ckpts/base_ndec2_log2/checkpoint-1000000")
ckpt_path = join(corpus_path, "ckpts/tiny_ndec1_log2/checkpoint-900000")
kenlm_model_path = "/home/randool/.pycorrector/datasets/zh_giga.no_cna_cmn.prune01244.klm"


def main():
    tokenizer = VCLSTokenizer.from_pretrained(ckpt_path)
    model = LogLMForPreTraining.from_pretrained(ckpt_path).to(device).eval()
    assert model.non_casual_decoder is not None
    model.casual_decoder = None

    # Load kenlm ngram model
    ngram = kenlm.LanguageModel(kenlm_model_path)

    num_lines = 100
    lines = []

    with open(test_file, encoding="utf8") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            lines.append(line)

    hit, total = 0, 1e-9
    loss, cnt = 0, 1e-9
    sentence_acc = []
    sentence_ppl = []

    with torch.no_grad(), tqdm(lines, ncols=80) as pbar:
        for line in pbar:
            inputs = tokenizer([line], return_tensors="pt", max_length=200, truncation=True).to(device)
            labels = inputs["labels"] = inputs["input_ids"]

            outputs = model.forward(**inputs)
            logits = outputs.ndec_prediction_logits

            pred = logits.argmax(dim=-1)
            hit_, length_ = pred.eq(labels).sum().item(), labels.shape[1]

            total += length_
            hit += hit_

            sentence_acc.append(hit_ / length_)
            sentence_ppl.append(ngram.perplexity(" ".join(line)))

            loss += outputs.loss.item()
            cnt += 1

            pbar.set_postfix({"acc": f"{(hit / total):.2%}", "loss": f"{(loss / cnt):.2e}"})

        pass

    plt.scatter(sentence_acc, sentence_ppl)
    plt.show()


if __name__ == "__main__":
    main()
