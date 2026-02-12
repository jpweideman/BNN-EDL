"""Hierarchical prior. Gaussian on weights and Gamma on Dirichlet strength."""

import torch
import torch.func as func
import posteriors
from src.registry import PRIOR_REGISTRY


@PRIOR_REGISTRY.register("dirichlet_strength")
class DirichletStrengthPrior:
    """Hierarchical prior. Gaussian weights and Gamma strength prior.
    
    Args:
        num_data: Number of training data points (for scaling)
        sd: Standard deviation for Gaussian weight prior (default: 1.0)
        a: Gamma shape parameter for strength prior (default: 2.0)
        b: Gamma rate parameter for strength prior (default: 0.1)
        normalize: Whether to normalize the Gaussian prior (default: False)
    """
    
    def __init__(self, num_data, sd=1.0, a=2.0, b=0.1, normalize=False):
        self.num_data = num_data
        self.sd = sd
        self.a = a
        self.b = b
        self.normalize = normalize
        self.model = None
    
    def __call__(self, params):
        """Compute hierarchical log prior.
        
        Args:
            params: Model parameters
        
        Returns:
            Log prior (scalar tensor)
        """
        # Gaussian prior on weights
        log_prior_weights = posteriors.diag_normal_log_prob(
            params,
            sd_diag=self.sd,
            normalize=self.normalize
        ) / self.num_data
        
        # Gamma prior on Dirichlet strength
        x, _ = self._batch
        device = next(iter(params.values())).device
        x = x.to(device)
        
        alpha = func.functional_call(self.model, params, x)
        S = alpha.sum(dim=-1)
        log_prior_strength = (self.a - 1) * torch.log(S) - self.b * S
        
        return log_prior_weights + log_prior_strength.mean()
