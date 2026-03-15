"""Dirichlet digamma likelihood for BNN-EDL classification."""

import torch
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("dirichlet_digamma")
class DirichletDigammaLikelihood:
    """Log-likelihood for BNN-EDL using expected cross-entropy (digamma).

    Computes E_{p ~ Dir(alpha)}[log p_y] = psi(alpha_y) - psi(S) instead of
    the marginal log alpha_y/S, providing a gradient signal for the evidence
    scale S.
    """

    def __call__(self, alpha, y_true):
        """Compute log-likelihood.

        Args:
            alpha: Dirichlet concentration parameters, shape [B, C]
            y_true: Ground truth labels, shape [B]

        Returns:
            Log-likelihood (scalar tensor), to be maximised by SGLD
        """
        S = alpha.sum(dim=-1)
        alpha_y = alpha[torch.arange(len(y_true)), y_true]
        log_likelihood = (torch.digamma(alpha_y) - torch.digamma(S)).mean()
        return log_likelihood
