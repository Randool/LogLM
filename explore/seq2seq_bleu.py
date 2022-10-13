"""
@File        :  seq2seq_bleu.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/13
@Version     :  1.0
@Description :  测试句子的生成结果
"""
import re
from os.path import expanduser, join
from typing import List

import torch
from sacrebleu import BLEU
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import DatasetWithIndex
from debug import dict_to_device
from model import LogLMForConditionalGeneration, LogLMTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
HOME = expanduser("~")
model_path = join(HOME, "Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/tiny_dec3_log2/checkpoint-1000000")
data_path = join(HOME, "Dataset/NLP/TencentMedicalPretrain/corpus/processed/dev/dev.data")


def corpus_bleu(hypotheses: List[str], references: List[str], lowercase=True, tokenize='zh',
                use_effective_order=False, max_ngram_order=4):
    metric = BLEU(
        lowercase=lowercase, tokenize=tokenize, effective_order=use_effective_order,
        max_ngram_order=max_ngram_order
    )
    return metric.corpus_score(hypotheses, references)


def strip_special_tokens(sent: str) -> str:
    sent = re.sub(r"\[(CLS.*?|SEP|PAD)]", "", sent)
    return "".join(sent.split())


def main():
    tokenizer: LogLMTokenizer = LogLMTokenizer.from_pretrained(model_path)
    model: LogLMForConditionalGeneration = LogLMForConditionalGeneration.from_pretrained(model_path).to(device)

    dataset = DatasetWithIndex(data_path, mask_token_id=tokenizer.mask_token_id,
                               pad_token_id=tokenizer.pad_token_id, log_base=tokenizer.log_base)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=dataset.collate_fn)

    hypotheses, references = [], []

    for inputs in tqdm(dataloader):
        labels = inputs.pop("labels")
        labels[labels == -100] = tokenizer.pad_token_id
        inputs["input_ids"] = labels
        inputs = dict_to_device(inputs, device)

        refers = list(map(strip_special_tokens, tokenizer.batch_decode(labels)))
        outputs = model.generate(**inputs, max_length=labels.shape[1] + 10)
        gens = list(map(strip_special_tokens, tokenizer.batch_decode(outputs)))

        hypotheses.extend(gens)
        references.extend(refers)

    print(corpus_bleu(hypotheses, [references], max_ngram_order=4))


if __name__ == "__main__":
    main()
