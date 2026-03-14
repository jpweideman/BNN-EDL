"""KL divergence utilities for EDL losses."""

import torch


def dirichlet_kl_divergence(alpha):
    """KL divergence KL[Dir(alpha) || Dir(1)].

    Args:
        alpha: Dirichlet parameters [B, C]

    Returns:
        KL divergence per sample [B, 1]
    """
    ones = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term


def annealed_kl(alpha, y, current_epoch, annealing_step):
    """KL divergence with linear annealing coefficient.

    Args:
        alpha: Dirichlet parameters [B, C]
        y: One-hot targets [B, C]
        current_epoch: Current training epoch
        annealing_step: Epoch at which coefficient reaches 1.0

    Returns:
        Annealed KL divergence per sample [B, 1]
    """
    anneal = min(1.0, current_epoch / annealing_step)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    return anneal * dirichlet_kl_divergence(kl_alpha)
