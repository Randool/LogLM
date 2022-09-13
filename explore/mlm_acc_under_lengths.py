"""
@File        :  mlm_acc_under_lengths.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/9
@Version     :  1.0
@Description :  测试不同长度下模型预测准确率
"""
from os.path import join, expanduser

import torch
from tqdm import tqdm

from model import LogLMTokenizer, LogLMForPreTraining


def calculate_acc_under_diff_len():
    corpus_path = join(expanduser("~"), "Dataset/NLP/TencentMedicalPretrain/corpus")
    test_file = join(corpus_path, "15.medical_plm_corpus/mfk.txt")
    ckpt_path = join(corpus_path, "ckpts/tiny_shared-false/checkpoint-1000000")

    lines = []
    num_lines = 100
    max_lengths = [7, 9, 16, 24, 28, 32, 58, 61, 64, 70, 71, 128, 192, 256]

    with open(test_file, encoding="utf8") as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            lines.append(line)

    lines.sort(key=lambda x: len(x))

    device = "cuda"
    tokenizer = LogLMTokenizer.from_pretrained(ckpt_path)
    model = LogLMForPreTraining.from_pretrained(ckpt_path).to(device).eval()

    for max_length in max_lengths:
        hit, total = 0, 1e-9
        loss, cnt = 0, 1e-9
        with torch.no_grad(), tqdm(lines, desc=f"len={max_length}", ncols=80) as pbar:
            for line in pbar:
                inputs = tokenizer([line], return_tensors="pt", max_length=max_length, truncation=True).to(device)
                inputs["mask_token_id"] = tokenizer.mask_token_id
                labels = inputs["labels"] = inputs["input_ids"]
                try:
                    outputs = model.forward(**inputs)
                except RuntimeError:
                    continue
                logits = outputs.ndec_prediction_logits
                pred = logits.argmax(dim=-1)
                total += labels.shape[1]
                hit += pred.eq(labels).sum().item()
                loss += outputs.loss.item()
                cnt += 1
                pbar.set_postfix({"acc": f"{(hit / total):.2%}", "loss": f"{(loss / cnt):.2e}"})


def main():
    calculate_acc_under_diff_len()


if __name__ == '__main__':
    main()
