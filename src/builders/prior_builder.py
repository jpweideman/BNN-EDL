"""Prior distribution builder for Bayesian inference."""

from src.builders.base import BaseBuilder
from src.registry import PRIOR_REGISTRY
import src.priors  # noqa: F401


class PriorBuilder(BaseBuilder):
    """Builds prior distributions for BNN posterior computation."""
    
    def build(self, num_data):
        """
        Build a prior function from configuration.
        
        Args:
            num_data: Dataset size (needed for scaling)
        
        Returns:
            Prior function instance 
        """
        prior_cls = PRIOR_REGISTRY.get(self.config.name)
        params = {k: v for k, v in self.config.items() if k != 'name'}
        prior_fn = prior_cls(num_data=num_data, **params)
        return prior_fn

