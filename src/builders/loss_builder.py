"""Loss function builder."""

from src.builders.base import BaseBuilder
from src.registry import LOSS_REGISTRY
import src.losses  # noqa: F401 # Triggers registration


class LossBuilder(BaseBuilder):
    """Builds loss functions from configuration."""
    
    def build(self):
        """
        Build a loss function from configuration.
        
        Returns:
            Loss function instance
        """
        loss_cls = LOSS_REGISTRY.get(self.config.name)
        params = getattr(self.config, 'params', {})
        loss_fn = loss_cls(**params)
        return loss_fn

