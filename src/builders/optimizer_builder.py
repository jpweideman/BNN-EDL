"""Optimizer builder."""

from src.builders.base import BaseBuilder
from src.registry import OPTIMIZER_REGISTRY
import src.optimizers  # noqa: F401 # Triggers registration


class OptimizerBuilder(BaseBuilder):
    """Builds optimizers from configuration."""
    
    def build(self, model_parameters, model=None, loss_fn=None, likelihood_fn=None, prior_fn=None):
        """
        Build optimizer from configuration.
        
        Args:
            model_parameters: Model parameters to optimize
            model: Full model (needed for BNN optimizers)
            loss_fn: Loss function (needed for standard optimizers)
            likelihood_fn: Likelihood function (needed for BNN optimizers)
            prior_fn: Prior function (needed for BNN optimizers)
        
        Returns:
            Optimizer instance
        """
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.config.name)
        params = {k: v for k, v in self.config.items() if k != 'name'}
        
        return optimizer_cls(
            model_parameters,
            model=model,
            loss_fn=loss_fn,
            likelihood_fn=likelihood_fn,
            prior_fn=prior_fn,
            **params
        )

