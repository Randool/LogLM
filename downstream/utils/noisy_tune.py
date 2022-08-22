import torch
import torch.nn as nn


def noisy_tune(model: nn.Module, noise_lambda: float) -> nn.Module:
    """
    NoisyTune: A Little Noise Can Help You Finetune Pretrained Language Models Better.
    https://aclanthology.org/2022.acl-short.76.pdf
    In general, setting λ to 0.1 ~ 0.15 can reach better result.
    """
    for name, param in model.named_parameters():
        model.state_dict()[name][:] += torch.randn(param.size()) * noise_lambda * torch.std(param)
    return model
