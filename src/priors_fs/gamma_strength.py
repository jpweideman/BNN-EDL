"""Gamma function-space prior on Dirichlet strength (total concentration)."""

import torch
from torch.distributions import Gamma
from src.registry import PRIORS_FS_REGISTRY


@PRIORS_FS_REGISTRY.register("gamma_strength")
class GammaStrengthPrior:
    """Gamma function-space prior on Dirichlet strength (sum of all class concentrations).

    Args:
        concentration:    Shape parameter (> 1). Prior mode at (a-1)/rate.
        rate:             Rate parameter (> 0).
        annealing_epochs: Linearly anneal weight from 0 to 1 over this many epochs.
    """

    def __init__(self, concentration: float, rate: float, annealing_epochs: int = 0):
        self.concentration = concentration
        self.rate = rate
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

    @property
    def weight(self) -> float:
        return 1.0 if self.annealing_epochs == 0 else min(self.current_epoch / self.annealing_epochs, 1.0)

    def __call__(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute log prior.

        Args:
            alpha: Dirichlet parameters, shape [B, C]

        Returns:
            Log prior (scalar tensor), weighted by annealing schedule.
        """
        alpha_0 = alpha.sum(dim=-1)
        dist = Gamma(
            torch.tensor(self.concentration, device=alpha_0.device),
            torch.tensor(self.rate, device=alpha_0.device),
        )
        return self.weight * dist.log_prob(alpha_0).mean()
