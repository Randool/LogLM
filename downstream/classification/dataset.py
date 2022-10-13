"""
@File        :  data.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/5/12
@Version     :  1.0
@Description :  Dataset for sentence classification tasks
"""
import json
from os.path import join
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

"""
CHIP-CTC: id, text[, label]
CHIP-STS: id, text1, text2, category[, label]
KUAKE-QIC: id, query[, label]
KUAKE-QTR: id, query, title[, label]
KUAKE-QQR: id, query1, query2[, label]
"""
task_text_fields = {
    "CHIP-CTC": ["text"],
    "CHIP-STS": ["category", "text1", "text2"],
    "KUAKE-QIC": ["query"],
    "KUAKE-QTR": ["query", "title"],
    "KUAKE-QQR": ["query1", "query2"],
}
task_id2label = {
    "CHIP-CTC": ['Disease', 'Multiple', 'Therapy or Surgery', 'Consent', 'Diagnostic', 'Laboratory Examinations',
                 'Pregnancy-related Activity', 'Age', 'Pharmaceutical Substance or Drug', 'Risk Assessment',
                 'Allergy Intolerance', 'Enrollment in other studies', 'Researcher Decision',
                 'Compliance with Protocol', 'Organ or Tissue Status', 'Sign', 'Addictive Behavior', 'Life Expectancy',
                 'Capacity', 'Symptom', 'Neoplasm Status', 'Device', 'Special Patient Characteristic',
                 'Non-Neoplasm Disease Stage', 'Data Accessible', 'Encounter', 'Diet', 'Smoking Status', 'Literacy',
                 'Oral related', 'Healthy', 'Address', 'Blood Donation', 'Gender', 'Receptor Status', 'Nursing',
                 'Sexual related', 'Exercise', 'Education', 'Disabilities', 'Bedtime', 'Alcohol Consumer',
                 'Ethical Audit', 'Ethnicity'],
    "CHIP-STS": ["0", "1"],
    "KUAKE-QIC": ['治疗方案', '其他', '病情诊断', '注意事项', '疾病表述', '就医建议',
                  '功效作用', '后果表述', '医疗费用', '病因分析', '指标解读'],
    "KUAKE-QTR": ["0", "1", "2", "3"],
    "KUAKE-QQR": ["0", "1", "2"],
}
task_label2id = {key: {label: i for i, label in enumerate(id2label)} for key, id2label in task_id2label.items()}


class SentClassificationDataset(Dataset):
    examples: List[dict]

    def __init__(self, cblue_root: str, task_name: str, mode: str):
        self.task_name = task_name
        self.mode = mode

        self.text_fields = task_text_fields[task_name]
        self.label2id = task_label2id[task_name]

        with open(join(cblue_root, task_name, f"{task_name}_{mode}.json"), encoding="utf8") as f:
            self.examples: List[dict] = json.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        texts = [self.examples[idx][field] for field in self.text_fields]
        if self.mode == "test":
            return texts, None
        label_id = self.label2id[self.examples[idx]["label"]]
        return texts, label_id


class CollateFn:
    def __init__(self, tokenizer: BertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[tuple]):
        if len(batch[0][0]) == 1:
            texts = [x[0][0] for x in batch]
        else:
            texts = [x[0] for x in batch]
            texts = [self.tokenizer.sep_token.join(x) for x in texts]  # AAA[SEP]BBB[SEP]CCC

        inputs = self.tokenizer(texts, padding="longest", truncation=True,
                                max_length=self.max_length, return_tensors="pt")

        if batch[0][1] is not None:
            inputs["labels"] = torch.tensor([x[1] for x in batch], dtype=torch.long)

        return dict(inputs)


def gather_labels():
    from debug_path import CBLUE_PATH
    from collections import Counter

    def _gather(name: str):
        label_counter = Counter()

        for mode in ["train", "dev"]:
            dataset = SentClassificationDataset(CBLUE_PATH, name, mode)
            for example in dataset.examples:
                label_counter[example["label"]] += 1

        print(name)
        pairs = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
        labels = [pair[0] for pair in pairs]
        print(labels)
        print()

    for task_name in task_id2label.keys():
        _gather(task_name)


def debug():
    from debug_path import CBLUE_PATH, BERT_ZH_PATH
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast
    from downstream.utils.useful import inputs_to_device

    tokenizer = BertTokenizerFast.from_pretrained(BERT_ZH_PATH, do_lower_case=True)
    for task_name in task_text_fields.keys():
        print(task_name)
        dataset = SentClassificationDataset(CBLUE_PATH, task_name, "dev")
        for inputs in DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=CollateFn(tokenizer, 64)):
            print(type(inputs))
            print(inputs_to_device(inputs, "cuda"))
            print(tokenizer.batch_decode(inputs["input_ids"]))
            break
        print()


if __name__ == '__main__':
    # gather_labels()
    debug()
