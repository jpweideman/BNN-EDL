"""Dirichlet-multinomial count MAP loss for deterministic eBNN training.

The count NLL plus the negative Gamma log-density of alpha_0 - the same terms
the sampler's log posterior carries, so SGD on this loss finds the MAP of the
eBNN posterior. Keep the Gamma parameters equal to training.prior_fs.

Reference for carrying the regularizer inside the loss: the EDL losses.
"""

import torch
from torch.distributions import Gamma
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dirichlet_multinomial_count_map")
class DirichletMultinomialCountMAP:
    """Dirichlet-multinomial count NLL plus the Gamma strength penalty on alpha_0.

    Args:
        concentration: Gamma shape parameter (> 1). Prior mode at (a-1)/rate.
        rate: Gamma rate parameter (> 0).
    """

    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate

    def __call__(self, alpha, y_true):
        alpha0 = alpha.sum(dim=-1)
        totals = y_true.sum(dim=-1)
        log_coeff = torch.lgamma(totals + 1) - torch.lgamma(y_true + 1).sum(dim=-1)
        log_dm = (log_coeff + torch.lgamma(alpha0) - torch.lgamma(totals + alpha0)
                  + (torch.lgamma(y_true + alpha) - torch.lgamma(alpha)).sum(dim=-1))
        log_gamma = Gamma(
            torch.tensor(self.concentration, dtype=alpha0.dtype, device=alpha0.device),
            torch.tensor(self.rate, dtype=alpha0.dtype, device=alpha0.device),
        ).log_prob(alpha0)
        return -(log_dm + log_gamma).mean()
