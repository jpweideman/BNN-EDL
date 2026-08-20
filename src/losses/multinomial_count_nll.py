"""Multinomial count NLL loss."""

import torch
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("multinomial_count_nll")
class MultinomialCountNLL:
    """Negative mean multinomial log probability of count vectors."""

    def __call__(self, y_pred, y_true):
        """Compute loss.

        Args:
            y_pred: Model predictions (logits), shape [B, C]
            y_true: Count vectors, shape [B, C]

        Returns:
            Loss value (scalar tensor)
        """
        log_probs = torch.log_softmax(y_pred, dim=-1)
        log_coeff = torch.lgamma(y_true.sum(dim=-1) + 1) - torch.lgamma(y_true + 1).sum(dim=-1)
        return -(log_coeff + (y_true * log_probs).sum(dim=-1)).mean()
