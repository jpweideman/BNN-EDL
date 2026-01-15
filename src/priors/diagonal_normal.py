"""Diagonal Normal prior for BNN parameters."""

import posteriors
from src.registry import PRIOR_REGISTRY


@PRIOR_REGISTRY.register("diagonal_normal")
class DiagonalNormalPrior:
    """
    Independent Gaussian prior on each parameter with mean 0 and standard deviation sd. 
    Returns scaled log prior.
    """
    
    def __init__(self, sd, num_data, normalize=False):
        self.sd = sd
        self.num_data = num_data
        self.normalize = normalize
    
    def __call__(self, params):
        log_prior = posteriors.diag_normal_log_prob(
            params,
            sd_diag=self.sd,
            normalize=self.normalize
        ) / self.num_data
        
        return log_prior

