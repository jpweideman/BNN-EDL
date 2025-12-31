"""Optimizer builder."""

from src.builders.base import BaseBuilder
from src.registry import OPTIMIZER_REGISTRY
import src.optimizers  # noqa: F401 # Triggers registration


class OptimizerBuilder(BaseBuilder):
    """Builds optimizers from configuration."""
    
    def build(self, model_parameters):
        """
        Build optimizer from configuration.
        
        Args:
            model_parameters: Model parameters to optimize
        
        Returns:
            PyTorch optimizer instance
        """
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.config.name)
        params = {k: v for k, v in self.config.items() if k != 'name'}
        return optimizer_cls(model_parameters, **params).optimizer

