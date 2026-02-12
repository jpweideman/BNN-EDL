"""Diagonal Normal prior for BNN parameters."""

import posteriors
from src.registry import PRIOR_REGISTRY


@PRIOR_REGISTRY.register("diagonal_normal")
class DiagonalNormalPrior:
    """Independent Gaussian prior on each parameter.
    
    Args:
        num_data: Number of training data points (for scaling)
        sd: Standard deviation for Gaussian prior (default: 1.0)
        normalize: Whether to normalize the prior (default: False)
    """
    
    def __init__(self, num_data, sd=1.0, normalize=False):
        self.sd = sd
        self.num_data = num_data
        self.normalize = normalize
    
    def __call__(self, params):
        """Compute log prior.
        
        Args:
            params: Model parameters
        
        Returns:
            Log prior (scalar tensor)
        """
        log_prior = posteriors.diag_normal_log_prob(
            params,
            sd_diag=self.sd,
            normalize=self.normalize
        ) / self.num_data
        
        return log_prior

