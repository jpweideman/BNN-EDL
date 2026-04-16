"""Joint Dirichlet-Categorical likelihood and Gibbs sampler for block-coordinate inference."""

import torch
from torch.distributions import Gamma
from src.registry import LIKELIHOOD_REGISTRY


def gibbs_sample_f(alpha: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Sample the exact Gibbs conditional p(f | w, y) via Dir-Cat conjugacy.

    Adds 1 to the true class concentration and samples from the resulting
    Dirichlet. Must be called inside torch.no_grad() with detached alpha.

    Args:
        alpha:   concentration parameters [B, C], alpha_c >= 1, detached
        y_true:  class labels [B], dtype long

    Returns:
        f_samples: shape [B, C], each row sums to 1
    """
    device = alpha.device
    alpha_posterior = alpha.cpu().clone()
    alpha_posterior[torch.arange(len(y_true)), y_true.cpu()] += 1.0
    g = Gamma(alpha_posterior, torch.ones_like(alpha_posterior)).sample()
    f = g / g.sum(dim=-1, keepdim=True)
    return f.to(device)


@LIKELIHOOD_REGISTRY.register("dirichlet_joint")
class DirichletJointLikelihood:
    """Log joint likelihood for Dirichlet-parameterized classification.

    Evaluates the Dirichlet log-density at a Gibbs sample of f, treating
    f as fixed observed data. Provides a gradient signal for the total
    concentration alpha_0 that the marginalized likelihood lacks.
    Compatible with any gradient-based optimizer, not specific to SGLD.
    """

    def __call__(self, alpha, f_samples):
        """Compute log-likelihood.

        Args:
            alpha:     Dirichlet concentration parameters, shape [B, C]
            f_samples: Gibbs samples drawn from p(f | w, y), shape [B, C], detached

        Returns:
            Log-likelihood (scalar tensor)
        """
        alpha_0 = alpha.sum(dim=-1)
        log_normalizer = torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=-1)  # [B]
        log_kernel = ((alpha - 1.0) * f_samples.log()).sum(dim=-1)                # [B]
        return (log_normalizer + log_kernel).mean()
