"""Likelihood function builder for Bayesian inference."""

from src.builders.base import BaseBuilder
from src.registry import LIKELIHOOD_REGISTRY
import src.likelihoods  # noqa: F401 # Triggers registration


class LikelihoodBuilder(BaseBuilder):
    """Builds likelihood functions for BNN posterior computation."""
    
    def build(self):
        """
        Build a likelihood function from configuration.
        
        Returns:
            Likelihood function instance 
        """
        likelihood_cls = LIKELIHOOD_REGISTRY.get(self.config.name)
        params = getattr(self.config, 'params', {})
        likelihood_fn = likelihood_cls(**params)
        return likelihood_fn

