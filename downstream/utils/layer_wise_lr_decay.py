"""
@File    :   lr_decay.py    
@Contact :   randool@sjtu.edu.cn
@Author  :   Randool
@Modify Time      @Version    @Description
------------      --------    -----------
2022/2/24 下午7:18   1.1         None
"""
import re
from typing import Tuple, Type, List, Dict

import torch.nn as nn
from transformers import BertLayer, Adafactor, AdamW
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer
from transformers.models.bert.modeling_bert import BertEmbeddings


def _layer_wise_traverse(model: nn.Module, stop_layers: Tuple[Type[nn.Module]] = ()):
    """
    Travel across layers, from encoder to decoder.
    Recursion is stop when meeting layers exist in `stop_layers`
    """
    layers = []

    for name, child in model.named_children():
        if isinstance(child, stop_layers):
            layers.append(child)
            continue
        layers += _layer_wise_traverse(child, stop_layers)

    if len(layers) == 0:
        layers.append(model)

    return layers


def config_layer_wise_lr(
        model: nn.Module, max_lr: float, max_weight_decay: float, decay_rate: float = 0.9,
        stop_layers: tuple = (BertLayer, BertEmbeddings),
        name_without_weight_decay: Tuple[str] = ("bias", "LayerNorm"),
        custom_layer_lr: Dict[int, float] = None,
) -> List[dict]:
    """ Specify per-layer learning rate

    Args:
        model:              The model
        max_lr:             Learning rate of the final layer in the model.
        max_weight_decay:   Weight decay of the final layer in the model.
        decay_rate:         Decay rate of both lr and weight_decay.
        stop_layers:        Stop recursive layers type
        name_without_weight_decay:    Likes: 'bias', 'LayerNorm', ...
        custom_layer_lr:    Manually configured learning rate for layers, id(layer) -> lr
    -------
    Examples::
        >>> from transformers import BertModel, BertLayer
        >>> from transformers.models.bert.modeling_bert import BertEmbeddings

        >>> ner_model: BertModel = BertModel.from_pretrained("xxx")
        >>> params = config_layer_wise_lr(ner_model,max_lr=5e-5,max_weight_decay=1e-6,decay_rate=0.9,stop_layers=(BertLayer, BertEmbeddings, TokenClassifier),custom_layer_lr={id(ner_model.classifier): 3e-5})
    """
    layers = _layer_wise_traverse(model, stop_layers)

    if len(name_without_weight_decay):
        no_decay_pattern = re.compile("|".join(name_without_weight_decay))
    else:
        no_decay_pattern = None

    lr = max_lr / decay_rate
    weight_decay = max_weight_decay / decay_rate

    param_list = []
    param_ids = set()   # Prevent the same parameter from setting different learning rates

    for layer in layers[::-1]:
        group_with_weight_decay = []
        group_without_weight_decay = []

        for name, param in layer.named_parameters():
            if id(param) in param_ids:
                continue
            param_ids.add(id(param))
            if (no_decay_pattern is not None) and (no_decay_pattern.search(name) is not None):
                group_without_weight_decay.append(param)
            else:
                group_with_weight_decay.append(param)

        if len(group_with_weight_decay) + len(group_without_weight_decay):
            # Config custom lr
            if custom_layer_lr is not None and id(layer) in custom_layer_lr:
                _lr = custom_layer_lr[id(layer)]
                _weight_decay = min(max_weight_decay, _lr * (max_weight_decay / max_lr))
            else:
                lr *= decay_rate
                weight_decay *= decay_rate
                _lr, _weight_decay = lr, weight_decay

            if len(group_with_weight_decay):
                param_list.append({
                    "params": group_with_weight_decay, "lr": _lr, "weight_decay": _weight_decay
                })
            if len(group_without_weight_decay):
                param_list.append({
                    "params": group_without_weight_decay, "lr": _lr, "weight_decay": 0.0
                })

    return param_list


def config_optimizer(
        train_args,
        model: nn.Module,
        decay_rate: float = 1.0,
        stop_layers: tuple = (BertLayer, BertEmbeddings, BartEncoderLayer, BartDecoderLayer),
        name_without_weight_decay: Tuple[str] = ("bias", "LayerNorm"),
        custom_layer_lr: Dict[int, float] = None
):
    """
    Args:
        train_args:         Training args should contain at least: {learning_rate, weight_decay}
        model:              The model
        decay_rate:         Learning rate of the model
        stop_layers:        Stop recursive layers type
        name_without_weight_decay:    Likes: 'bias', 'LayerNorm', ...
        custom_layer_lr:    Manually configured learning rate for layers, id(layer) -> lr
    """
    optimizer_grouped_parameters = config_layer_wise_lr(
        model=model, max_lr=train_args.learning_rate,
        max_weight_decay=train_args.weight_decay,
        decay_rate=decay_rate,
        stop_layers=stop_layers,
        name_without_weight_decay=name_without_weight_decay,
        custom_layer_lr=custom_layer_lr
    )
    if getattr(train_args, "adafactor", False):
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (getattr(train_args, "adam_beta1", 0.9), getattr(train_args, "adam_beta2", 0.999)),
            "eps": getattr(train_args, "adam_epsilon", 1e-6),
        }
    optimizer_kwargs["lr"] = train_args.learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
