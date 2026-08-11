"""Gamma function-space prior on Dirichlet strength (total concentration)."""

import warnings
import torch
from torch.distributions import Gamma
from src.registry import PRIORS_FS_REGISTRY


@PRIORS_FS_REGISTRY.register("gamma_strength")
class GammaStrengthPrior:
    """Gamma function-space prior on Dirichlet strength (sum of all class concentrations).

    Args:
        concentration:    Shape parameter (> 1). Prior mode at (a-1)/rate.
        rate:             Rate parameter (> 0).
        num_classes:      Number of output classes; used to validate that the prior mode > num_classes.
        annealing_epochs: Linearly anneal weight from 0 to 1 over this many epochs.
    """

    def __init__(self, concentration: float, rate: float, num_classes: int, annealing_epochs: int = 0):
        self.concentration = concentration
        self.rate = rate
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

        if concentration <= 1:
            warnings.warn(
                f"GammaStrengthPrior: concentration={concentration} <= 1, so the Gamma has no "
                f"interior mode (mode=0). The prior cannot pin alpha_0 > {num_classes}.",
                stacklevel=2,
            )
        elif self.mode <= num_classes:
            warnings.warn(
                f"GammaStrengthPrior: prior mode = (concentration-1)/rate = "
                f"({concentration}-1)/{rate} = {self.mode:.4g} <= num_classes={num_classes}. "
                f"The Dirichlet requires alpha_0 > {num_classes} for a valid interior mode.",
                stacklevel=2,
            )

    @property
    def mode(self) -> float:
        """Prior mode of alpha_0 (0 when the Gamma has no interior mode)."""
        return max(self.concentration - 1.0, 0.0) / self.rate

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
