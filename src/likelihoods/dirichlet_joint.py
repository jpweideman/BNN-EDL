"""Joint Dirichlet-Categorical log-likelihood for block-coordinate inference."""

import torch
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("dirichlet_joint")
class DirichletJointLikelihood:
    """Log joint likelihood for Dirichlet-parameterized classification.

    Evaluates the Dirichlet log-density at a Gibbs sample of f, treating
    f as fixed observed data. Provides a gradient signal for the total
    concentration alpha_0 that the marginalized likelihood lacks.
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
