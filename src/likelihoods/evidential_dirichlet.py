"""Evidential Dirichlet pseudo-likelihood for EDL + SGLD."""

import torch
import torch.nn.functional as F
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("evidential_dirichlet")
class EvidentialDirichletLikelihood:
    """Pseudo-Log-Likelihood for Evidential Deep Learning combined with SGLD.
    
    Replaces the standard scale-invariant log likelihood with the Digamma 
    expected log-likelihood and a KL-divergence penalty.
    """

    def __init__(self, num_classes, max_annealing_steps):
        self.num_classes = num_classes
        self.max_annealing_steps = max_annealing_steps
        self.current_step = 0

    def __call__(self, alpha, y_true):
        """
        Args:
            alpha: Dirichlet parameters, shape [B, C]
            y_true: Ground truth labels, shape [B]

        Returns:
            Pseudo-log-likelihood scalar to be maximized by SGLD.
        """
        y_onehot = F.one_hot(y_true, num_classes=self.num_classes).float()
        S = alpha.sum(dim=-1, keepdim=True)

        expected_log_lik = (y_onehot * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1)

        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        kl_term = (
            torch.lgamma(S_tilde)
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=-1, keepdim=True)
            - torch.lgamma(torch.tensor(self.num_classes, dtype=alpha.dtype, device=alpha.device))
        ).squeeze(-1)

        annealing_coef = min(1.0, self.current_step / self.max_annealing_steps)
        self.current_step += 1

        return (expected_log_lik - annealing_coef * kl_term).mean()
