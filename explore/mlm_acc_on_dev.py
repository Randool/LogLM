"""
@File        :  mlm_acc_on_dev.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/9/9
@Version     :  1.0
@Description :  计算MLM的验证集整体正确率
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DatasetForShuffledFile
from model import LogLMTokenizer, LogLMForPreTraining
from plm import PlmForPreTraining, ChPTMTokenizer


def dict_to_device(inputs: dict, device):
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs


def calculate_mlm_acc_on_dev():
    is_loglm = True
    device = "cuda"
    loglm_path = "/home/randool/Dataset/NLP/TencentMedicalPretrain/corpus/ckpts/base_ndec2_log1024/checkpoint-800000"

    if is_loglm:
        model = LogLMForPreTraining.from_pretrained(loglm_path).eval().to(device)
        log_tokenizer = tokenizer = LogLMTokenizer.from_pretrained(loglm_path)
    else:
        model_path = "/home/randool/Dataset/transformers/our-macbert-base-1000000"
        model = PlmForPreTraining.from_pretrained(model_path).eval().to(device)
        tokenizer = ChPTMTokenizer.from_pretrained(model_path)
        log_tokenizer = LogLMTokenizer.from_pretrained(loglm_path)

    devset = DatasetForShuffledFile(data_file="/home/randool/tmp/dev.data", mask_token_id=log_tokenizer.mask_token_id,
                                    pad_token_id=log_tokenizer.pad_token_id, log_base=1024)
    dataloader = DataLoader(devset, batch_size=1, collate_fn=devset.collate_fn)

    masked_cnt = 0
    correct_cnt = 0

    with torch.no_grad(), tqdm(dataloader) as pbar:
        for inputs in pbar:
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            mask_ids = input_ids.eq(tokenizer.mask_token_id)

            if is_loglm:
                mlm_pred = model(**dict_to_device(inputs, device)).prediction_logits.argmax(dim=-1).cpu()
            else:
                masked_sent = log_tokenizer.decode(input_ids)
                sent = log_tokenizer.decode(labels)

                input_ids = tokenizer.encode(masked_sent, add_special_tokens=False, return_tensors="pt").to(device)
                labels = tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
                if input_ids.shape != labels.shape:
                    continue
                # assert input_ids.shape == labels.shape, f"input_ids: {input_ids.shape}, labels: {labels.shape}"
                mlm_pred = model(input_ids=input_ids).prediction_logits.argmax(dim=-1).cpu()

            correct_cnt += mlm_pred[mask_ids].eq(labels[mask_ids]).sum().item()
            masked_cnt += mask_ids.sum().item()

            pbar.set_postfix({"acc": f"{(correct_cnt / masked_cnt):.2%}"})


def main():
    calculate_mlm_acc_on_dev()


if __name__ == '__main__':
    main()
