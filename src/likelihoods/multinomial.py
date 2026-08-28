"""Multinomial likelihood for BNN count prediction."""

import torch
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("multinomial")
class MultinomialLikelihood:
    """Log-likelihood of count vectors under Mult(R, softmax(logits))."""

    def __call__(self, y_pred, y):
        """Compute log-likelihood.

        Args:
            y_pred: Model predictions (logits), shape [B, C]
            y: Count vectors, shape [B, C]

        Returns:
            Log-likelihood (scalar tensor)
        """
        log_probs = torch.log_softmax(y_pred, dim=-1)
        log_coeff = torch.lgamma(y.sum(dim=-1) + 1) - torch.lgamma(y + 1).sum(dim=-1)
        log_likelihood = (log_coeff + (y * log_probs).sum(dim=-1)).mean()
        return log_likelihood
