"""EDL MSE loss for evidential deep learning.

Reference:
    https://github.com/dougbrion/pytorch-classification-uncertainty/blob/master/losses.py
"""

import torch.nn.functional as F
from src.registry import LOSS_REGISTRY
from src.losses.utils.kl import annealed_kl


@LOSS_REGISTRY.register("edl_mse")
class EDLMSELoss:
    """EDL Bayes risk under sum-of-squares loss (equation 5, Sensoy et al. 2018).

    Args:
        num_classes: Number of output classes
        annealing_step: Epoch at which KL coefficient reaches 1.0
    """

    def __init__(self, num_classes, annealing_step):
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.current_epoch = 0

    def __call__(self, output, target):
        y = F.one_hot(target, self.num_classes).float()
        alpha = output
        S = alpha.sum(dim=1, keepdim=True)
        err = ((y - alpha / S) ** 2).sum(dim=1, keepdim=True)
        var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)
        return (err + var + annealed_kl(alpha, y, self.current_epoch, self.annealing_step)).mean()
