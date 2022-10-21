"""
@File        :  main.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/5/12
@Version     :  1.0
@Description :  Sentence classification tasks
"""
import copy
import json
import pickle
from glob import glob
from os.path import join
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import set_seed, BertLayer, BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers.models.bert.modeling_bert import BertEmbeddings

from downstream.classification.dataset import SentClassificationDataset, CollateFn, task_id2label
from model import LogLMForSequenceClassification, VCLSTokenizer
from plm import ChPTMTokenizer, PlmForSequenceClassification
from downstream.utils.args import ModelConstructArgs, ClassifyDataArgs
from utils.callback import LoggerCallback
from downstream.utils.can import can
from downstream.utils.layer_wise_lr_decay import config_optimizer
from downstream.utils.noisy_tune import noisy_tune
from downstream.utils.rdrop_trainer import RDropTrainer
from downstream.utils.useful import best_model_checkpoint, get_logger_and_args, k_fold_recombine, softmax


def get_model_with_tokenizer(model_args: ModelConstructArgs, task_name: str):
    MODEL_TOKENIZER = {
        "plm": (PlmForSequenceClassification, ChPTMTokenizer),
        "bert": (BertForSequenceClassification, BertTokenizer),
        "loglm": (LogLMForSequenceClassification, VCLSTokenizer),
    }
    _M, _T = MODEL_TOKENIZER[model_args.model_type]

    if _T == VCLSTokenizer:
        tokenizer = _T.from_pretrained(model_args.model_path, do_lower_case=True,
                                       anchored_length=model_args.anchored_cls_length)
        model = _M.from_pretrained(model_args.model_path, num_labels=len(task_id2label[task_name]),
                                   pooling_all=model_args.pooling_all)
    else:
        tokenizer = _T.from_pretrained(model_args.model_path, do_lower_case=True)
        model = _M.from_pretrained(model_args.model_path, num_labels=len(task_id2label[task_name]))

    return model, tokenizer


class ComputeMetrics:
    def __init__(self, logger, labels: list, output_dir: str):
        self.cnt = 0
        self.logger = logger
        self.labels = labels
        self.output_dir = output_dir

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def __call__(self, eval_pred) -> dict:
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)  # shape: [batch, num_labels]
        f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")

        try:
            mtx = confusion_matrix(labels, predictions, labels=self.labels)  # Calculate confusion matrix
            confusion_log = f"Confusion Matrix:\nLabels:{self.labels}\n{mtx}"
            self.logger.info(confusion_log)

            C = mtx.shape[0]
            S = max(3., C / 5)
            fig, ax = plt.subplots(figsize=(S, S))
            disp = ConfusionMatrixDisplay(mtx, display_labels=self.labels)
            disp.plot(ax=ax)
            plt.tight_layout()
            plt.savefig(join(self.output_dir, f"confusion_mtx_{self.cnt}.png"))
            self.cnt += 1

            # Do `Classification with Alternating Normalization`
            counter = np.unique(labels, return_counts=True)[1]
            normalized_p = can(prior_cnt=counter, logits=logits, k=min(C, 3))
            mtx_ = confusion_matrix(labels, normalized_p.argmax(axis=-1))
            f1_ = f1_score(y_true=labels, y_pred=predictions, average="macro")
            self.logger.info(f"Normalized:\n{mtx_}\nraw:{f1:.2%} ==> norm:{f1_:.2%}")

        except ValueError as e:  # 防止某些类别不出现在dev集中
            self.logger.info("Skip draw confusion matrix. ValueError: " + str(e))

        return {"f1": f1}


def data_augment(task_name: str, dataset: SentClassificationDataset, logger):
    if task_name == "CHIP-STS":
        more = []
        for example in dataset.examples:
            example = copy.deepcopy(example)
            example["text2"], example["text1"] = example["text1"], example["text2"]
            more.append(example)
        logger.info(f"DA: + {len(more)} examples")
        dataset.examples += more


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, (train_args, model_args, data_args) = get_logger_and_args(
        __name__, [TrainingArguments, ModelConstructArgs, ClassifyDataArgs], _args
    )
    assert data_args.task_name in ("CHIP-CTC", "CHIP-STS", "KUAKE-QIC", "KUAKE-QTR", "KUAKE-QQR"), \
        f"Unrecognized task_name: {data_args.task_name}"
    id2label = task_id2label[data_args.task_name]
    logger.info(f"Task: {data_args.task_name}, labels: {len(id2label)}")

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args, data_args.task_name)

    # ===== Get datasets =====
    if train_args.do_train:
        trainset = SentClassificationDataset(data_args.cblue_root, data_args.task_name, "train")
        devset = SentClassificationDataset(data_args.cblue_root, data_args.task_name, "dev")

        # Do k-fold cross validation
        if data_args.total_folds > 1:
            n_train, n_dev = len(trainset), len(devset)
            logger.info(f"[{data_args.curr_fold}] of {data_args.total_folds}-folds cross validation")
            trainset, devset = k_fold_recombine(trainset, devset, ["examples"],
                                                data_args.curr_fold, data_args.total_folds)
            logger.info(f"len(train): {n_train} -> {len(trainset)}")
            logger.info(f"len(dev): {n_dev} -> {len(devset)}")

        data_augment(data_args.task_name, trainset, logger)

        if data_args.extra_train_datas is not None:
            for extra_data in data_args.extra_train_datas:
                files = sorted(glob(extra_data))
                if len(files) == 0:
                    logger.warning("Cannot match any file with `extra_train_datas`")
                else:
                    for file in files:
                        with open(file, encoding="utf8") as f:
                            data = json.load(f)
                        trainset.examples += data
                        logger.info(f"++ {len(data)} train examples from {file}")

        logger.info(f"Trainset: {len(trainset)} samples")
        logger.info(f"Devset: {len(devset)} samples")

        # Noisy tune
        if model_args.noise_lambda > 0:
            model = noisy_tune(model, model_args.noise_lambda)

        optimizer = config_optimizer(
            train_args=train_args, model=model, decay_rate=model_args.layer_wise_lr_decay,
            stop_layers=(BertLayer, BertEmbeddings)
        )
    else:
        trainset, devset = None, None
        optimizer = None

    # ===== Trainer =====
    if data_args.use_rdrop:
        trainer = RDropTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFn(tokenizer, data_args.max_length),
            train_dataset=trainset,
            eval_dataset=devset,
            callbacks=[LoggerCallback(logger)],
            compute_metrics=ComputeMetrics(logger, id2label, train_args.output_dir),
            optimizers=(optimizer, None),
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFn(tokenizer, data_args.max_length),
            train_dataset=trainset,
            eval_dataset=devset,
            callbacks=[LoggerCallback(logger)],
            compute_metrics=ComputeMetrics(logger, id2label, train_args.output_dir),
            optimizers=(optimizer, None),
        )

    if train_args.do_train:
        logger.info(f"Device: {train_args.device}")
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")
            best_path = best_model_checkpoint(train_args.output_dir)
            trainer.model.from_pretrained(best_path).to(train_args.device)
            logger.info(f"Load best checkpoint from {best_path}")

    if train_args.do_predict:
        test_dataset = SentClassificationDataset(data_args.cblue_root, data_args.task_name, "test")
        logger.info(f"Testset: {len(test_dataset)} samples")

        if not (train_args.do_train and train_args.load_best_model_at_end):
            best_path = best_model_checkpoint(train_args.output_dir)
            trainer.model.from_pretrained(best_path).to(train_args.device)
            logger.info(f"Load best checkpoint from {best_path}")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")

        # Save raw predictions
        probability = softmax(predictions)  # shape: [batch, num_labels]
        with open(join(train_args.output_dir, "probability.pkl"), "wb") as f:
            pickle.dump(probability, f)
            logger.info(f"Save pickled probability to {join(train_args.output_dir, 'probability.pkl')}")

        # Save final results
        preds = predictions.argmax(axis=-1)

        assert len(test_dataset.examples) == len(preds), \
            f"Length mismatch, testset.examples: {len(test_dataset.examples)}, preds: {len(preds)}"

        for example, pred in zip(test_dataset.examples, preds):
            example["label"] = id2label[pred]

        with open(join(train_args.output_dir, data_args.test_output_file), "w", encoding="utf8") as f:
            json.dump(test_dataset.examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Save final results to {join(train_args.output_dir, data_args.test_output_file)}")


if __name__ == '__main__':
    main()
