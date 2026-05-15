"""Gamma function-space prior on the total Dirichlet concentration α₀."""

import torch
from torch.distributions import Gamma
from src.registry import PRIORS_FS_REGISTRY


@PRIORS_FS_REGISTRY.register("gamma_alpha0")
class GammaAlpha0Prior:
    """Gamma function-space prior on the total concentration alpha_0(sum over all class concentrations).

    Args:
        concentration: Shape parameter (> 1). Prior mode at (a-1)/rate.
        rate:          Rate parameter (> 0).
    """

    def __init__(self, concentration: float, rate: float):
        self.concentration = concentration
        self.rate = rate

    def __call__(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute log prior.

        Args:
            alpha: Dirichlet parameters, shape [B, C]

        Returns:
            Log prior (scalar tensor)
        """
        alpha_0 = alpha.sum(dim=-1)
        dist = Gamma(
            torch.tensor(self.concentration, device=alpha_0.device),
            torch.tensor(self.rate, device=alpha_0.device),
        )
        return dist.log_prob(alpha_0).mean()
