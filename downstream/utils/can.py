"""
@File        :  can.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2022/5/15
@Version     :  1.0
@Description :  Classification with Alternating Normalization, https://spaces.ac.cn/archives/8728
"""
import numpy as np

from downstream.utils.useful import softmax


def can(prior_cnt: np.ndarray, logits: np.ndarray, threshold: float = 0.9, k: int = -1, epoch: int = 1,
        inplace=False) -> np.ndarray:
    """ 用先验分布调整预测分布

    Args:
        prior_cnt:  先验数量分布 `[C]`，C表示类别数
        logits:     模型预测的概率分布 `[N, C]`
        threshold:  阈值
        k:          选取top-k个特征计算不确定性k<=C
        epoch:      迭代次数
        inplace:    在p_pred上修改
    Return:
        调整后的p_pred
    """
    p_pred = softmax(logits)

    C = p_pred.shape[-1]
    k = C if k == -1 else k
    assert k <= C, f"top-k({k}) should be less than or equal to C({C})"

    # 计算不确定性
    pred_topk = np.sort(p_pred, axis=1)[:, -k:]
    pred_topk /= pred_topk.sum(axis=1, keepdims=True)
    pred_uncertainty = -(pred_topk * np.log(pred_topk)).sum(axis=1) / np.log(k)

    pred_confident = p_pred[pred_uncertainty < threshold]

    if len(prior_cnt.shape) == 1:
        prior_cnt = prior_cnt[None]
    p_prior = prior_cnt / prior_cnt.sum()

    # 逐个调整置信度低的预测分布
    for coo in np.where(pred_uncertainty >= threshold)[0]:
        p_pred_ = np.concatenate([pred_confident, p_pred[coo][None]], axis=0)
        for i in range(epoch):
            p_pred_ = p_pred_ * p_prior / p_pred_.mean(axis=0, keepdims=True)  # 行间归一化
            p_pred_ /= p_pred_.sum(axis=1, keepdims=True)  # 行内归一化
        p_pred[coo] = p_pred_[-1]

    return p_pred
