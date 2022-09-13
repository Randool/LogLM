"""
@File        :  cls_info_storage.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/9
@Version     :  1.0
@Description :  测试信息在CLS中的存储方式
"""
import json
from functools import partial

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


def sub_last_hidden_states_hook(hidden_states, last_hidden_states):
    return last_hidden_states - hidden_states


def mask_some_of_cls(hidden_states, cls_len: int):
    reserved_len = cls_len // 2
    hidden_states[:, reserved_len:] = 0
    return hidden_states


def channel_mask(hidden_states):
    H = hidden_states.shape[-1]
    hidden_states[:, :, :H // 2] = 0
    return hidden_states


@torch.no_grad()
def main():
    device = "cuda"
    loglm_path = "/home/randool/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base_ndec2_log2/checkpoint-1000000"
    data_path = "/home/randool/Dataset/NLP/TencentMedicalPretrain/corpus/processed_mlm/dev/dev.data"

    model: LogLMForPreTraining = LogLMForPreTraining.from_pretrained(loglm_path).eval().to(device)
    tokenizer: LogLMTokenizer = LogLMTokenizer.from_pretrained(loglm_path)

    devset = DatasetForShuffledFile(data_file=data_path, mask_token_id=tokenizer.mask_token_id,
                                    pad_token_id=tokenizer.pad_token_id, log_base=2)
    dataloader = DataLoader(devset, batch_size=1, collate_fn=devset.collate_fn)

    correct_cnt = 0
    total_cnt = 1e-8

    with tqdm(dataloader) as pbar, open("cls_exp.txt", "w", encoding="utf8") as f:
        for i, inputs in enumerate(pbar):
            inputs = dict_to_device(inputs, device)
            result = []

            # First pass
            labels = inputs["input_ids"] = inputs["labels"]
            result.append(tokenizer.batch_decode(labels)[0])

            # hook = partial(mask_some_of_cls, cls_len=inputs["cls_type_ids"].sum().item())
            # hook = channel_mask
            # hook = lambda x: x
            hook = None
            outputs = model.forward(last_hidden_states_hook=hook, **inputs)

            # Second pass
            mask_pos = (1 - inputs["token_type_ids"]) - inputs["cls_type_ids"]
            masked_ids = labels[:]
            masked_ids[mask_pos == 1] = tokenizer.mask_token_id
            inputs["input_ids"] = masked_ids

            hook = partial(sub_last_hidden_states_hook, last_hidden_states=outputs.hidden_states)
            outputs2 = model.forward(last_hidden_states_hook=hook, **inputs)

            # Pred result
            mlm_pred = outputs2.ndec_prediction_logits.argmax(dim=-1)
            result.append(tokenizer.batch_decode(mlm_pred)[0])

            total_cnt += mlm_pred.numel()
            correct_cnt += (mlm_pred == labels).sum().item()

            f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

            if i >= 100:
                break

    print(f"acc: {correct_cnt / total_cnt:.2%}")


if __name__ == '__main__':
    main()
