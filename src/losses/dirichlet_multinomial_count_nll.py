"""Dirichlet-multinomial count NLL loss."""

import torch
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dirichlet_multinomial_count_nll")
class DirichletMultinomialCountNLL:
    """Negative mean Dirichlet-multinomial log probability of count vectors."""

    def __call__(self, y_pred, y_true):
        """Compute loss.

        Args:
            y_pred: Dirichlet concentrations, shape [B, C]
            y_true: Count vectors, shape [B, C]

        Returns:
            Loss value (scalar tensor)
        """
        alpha0 = y_pred.sum(dim=-1)
        totals = y_true.sum(dim=-1)
        log_coeff = torch.lgamma(totals + 1) - torch.lgamma(y_true + 1).sum(dim=-1)
        return -(log_coeff
                 + torch.lgamma(alpha0) - torch.lgamma(totals + alpha0)
                 + (torch.lgamma(y_true + y_pred) - torch.lgamma(y_pred)).sum(dim=-1)).mean()
