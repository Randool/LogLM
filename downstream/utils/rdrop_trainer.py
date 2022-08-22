"""
@File        :  rdrop_trainer.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/6/11
@Version     :  1.0
@Description :  None
"""
from typing import Dict, Union, Optional, Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, TrainerCallback, \
    EvalPrediction, TrainingArguments, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class RDropTrainer(Trainer):
    """
    Trainer that supports RDrop training. Reference: https://github.com/dropreg/R-Drop
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            tokenizer=tokenizer, model_init=model_init, compute_metrics=compute_metrics,
            callbacks=callbacks, optimizers=optimizers
        )

    def get_sample_weight(self, input_ids: torch.Tensor) -> List[float]:
        # TODO: 记录每个sample的权重，下一个epoch时使用
        batch_size = input_ids.shape[0]
        return [1] * batch_size

    def compute_loss(self, model: BertForSequenceClassification, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs: SequenceClassifierOutput = model.forward(**inputs)

        if outputs.loss is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        outputs2 = model.forward(**inputs)
        loss = outputs.loss + outputs2.loss

        # R-Drop
        logits1, logits2 = outputs.logits, outputs2.logits
        p, q = torch.log_softmax(logits1, dim=-1), torch.log_softmax(logits2, dim=-1)
        p_tec, q_tec = torch.softmax(logits1, dim=-1), torch.softmax(logits2, dim=-1)
        kl_loss = (F.kl_div(p, q_tec, reduction="sum") + F.kl_div(q, p_tec, reduction="sum")) / 2

        loss = loss + 0.5 * kl_loss
        outputs.loss = loss

        return (loss, outputs) if return_outputs else loss
