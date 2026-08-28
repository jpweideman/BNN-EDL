"""Dirichlet-multinomial likelihood for eBNN count prediction."""

import torch
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("dirichlet_multinomial")
class DirichletMultinomialLikelihood:
    """Log-likelihood of count vectors under DM(R, alpha)."""

    def __call__(self, alpha, y):
        """Compute log-likelihood.

        Args:
            alpha: Dirichlet concentrations, shape [B, C]
            y: Count vectors, shape [B, C]

        Returns:
            Log-likelihood (scalar tensor)
        """
        alpha0 = alpha.sum(dim=-1)
        totals = y.sum(dim=-1)
        log_coeff = torch.lgamma(totals + 1) - torch.lgamma(y + 1).sum(dim=-1)
        log_likelihood = (log_coeff
                          + torch.lgamma(alpha0) - torch.lgamma(totals + alpha0)
                          + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(dim=-1)).mean()
        return log_likelihood
