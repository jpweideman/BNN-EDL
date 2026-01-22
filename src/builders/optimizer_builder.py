"""Optimizer builder."""

from src.builders.base import BaseBuilder
from src.registry import OPTIMIZER_REGISTRY
from src.optimizers.bnn import BNNOptimizer
import src.optimizers  # noqa: F401 # Triggers registration


class OptimizerBuilder(BaseBuilder):
    """Builds optimizers from configuration."""
    
    def build(self, model_parameters, model=None, loss_fn=None, likelihood_fn=None, prior_fn=None, num_data=None):
        """
        Build optimizer from configuration.
        
        Args:
            model_parameters: Model parameters to optimize
            model: Full model (needed for BNN optimizers)
            loss_fn: Loss function (needed for standard optimizers)
            likelihood_fn: Likelihood function (needed for BNN optimizers)
            prior_fn: Prior function (needed for BNN optimizers)
            num_data: Number of training data points (needed for BNN optimizers)
        
        Returns:
            Optimizer instance
        """
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.config.name)
        params = {k: v for k, v in self.config.items() if k != 'name'}
        
        # Check if BNN optimizer using type checking
        is_bnn_optimizer = issubclass(optimizer_cls, BNNOptimizer)
        
        if is_bnn_optimizer:
            # BNN optimizers
            return optimizer_cls(**params)(
                model_parameters,
                model=model,
                likelihood_fn=likelihood_fn,
                prior_fn=prior_fn,
                num_data=num_data
            )
        else:
            # Standard optimizers
            return optimizer_cls(
                model_parameters,
                model=model,
                loss_fn=loss_fn,
                likelihood_fn=likelihood_fn,
                prior_fn=prior_fn,
                **params
            )

