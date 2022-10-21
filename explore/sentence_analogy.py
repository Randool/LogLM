"""
@File        :  sentence_analogy.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/9
@Version     :  1.0
@Description :  测试LogLM词向量表达对句子对关系的建模能力
"""
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from model import VCLSTokenizer, LogLMForConditionalGeneration
from wmd import WMD

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer_and_model(model_type: str):
    if model_type == "loglm":
        model_path = "/home/randool/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/loglm_base_dec6_log2/checkpoint-500000"
        tokenizer: VCLSTokenizer = VCLSTokenizer.from_pretrained(model_path)
        model: LogLMForConditionalGeneration = LogLMForConditionalGeneration.from_pretrained(model_path).eval().to(
            device)

    elif model_type == "bart":
        tokenizer = BertTokenizer.from_pretrained("model_path")
        model = BertModel.from_pretrained("model_path")

    else:
        raise ValueError

    return tokenizer, model


def load_sentence_pairs(data_path: str, count=-1):
    with open(data_path, encoding="utf8") as f:
        samples = json.load(f)
    samples = samples if count == -1 else samples[:count]
    for sample in tqdm(samples, total=len(samples), ncols=64):
        yield sample["text1"], sample["text2"], int(sample["label"])


@torch.no_grad()
def main():
    data_path = "/home/randool/Dataset/NLP/CBLUEDatasets/CHIP-STS/CHIP-STS_train.json"
    tokenizer, model = load_tokenizer_and_model("loglm")

    pred_labels = []

    for (text1, text2, label) in load_sentence_pairs(data_path, 1000):
        inputs = tokenizer([text1, text2], padding=True, return_tensors="pt").to(device)
        L1, L2 = inputs["cls_type_ids"].sum(dim=-1).tolist()
        cls_hidden_states = model.forward(**inputs).encoder_last_hidden_state
        cls1 = cls_hidden_states[0][:L1].cpu().numpy()
        cls2 = cls_hidden_states[1][:L2].cpu().numpy()
        embeds = np.concatenate([cls1, cls2], axis=0, dtype=np.float32)
        nbow = {
            "s1": ("#1", [i for i in range(L1)], np.ones(L1, dtype=np.float32) / L1),
            "s2": ("#2", [i + L1 for i in range(L2)], np.ones(L2, dtype=np.float32) / L2)
        }
        calc = WMD(embeds, nbow, vocabulary_min=2)
        pred_labels.append((calc.nearest_neighbors("s1")[0][1], label))

    pred_labels.sort(key=lambda x: x[0], reverse=True)

    cnt_one = 0
    integration = 0

    for i, pairs in enumerate(pred_labels, start=1):
        cnt_one += pairs[1]
        integration += cnt_one / i

    print(integration / len(pred_labels))


if __name__ == "__main__":
    main()
