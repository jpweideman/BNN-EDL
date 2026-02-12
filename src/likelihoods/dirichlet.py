"""Dirichlet likelihood for BNN classification with Dirichlet-parameterized output."""

import torch
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("dirichlet")
class DirichletLikelihood:
    """Log-likelihood for Dirichlet-parameterized classification."""
    
    def __call__(self, alpha, y_true):
        """Compute log-likelihood.
        
        Args:
            alpha: Dirichlet parameters, shape [B, C]
            y_true: Ground truth labels, shape [B]
        
        Returns:
            Log-likelihood (scalar tensor)
        """
        S = alpha.sum(dim=-1)
        alpha_y = alpha[torch.arange(len(y_true)), y_true]
        log_likelihood = torch.log(alpha_y / S).mean()
        return log_likelihood

