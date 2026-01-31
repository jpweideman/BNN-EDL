"""Output layer builder."""

from src.builders.base import BaseBuilder
from src.registry import OUTPUT_LAYER_REGISTRY
import src.models.output_layers  # Triggers registration


class OutputLayerBuilder(BaseBuilder):
    """Builds output layers from configuration."""
    
    def build(self, input_dim):
        """Build output layer from config.
        
        Args:
            input_dim: Input feature dimension from model backbone
            
        Returns:
            Output layer module
        """
        layer_cls = OUTPUT_LAYER_REGISTRY.get(self.config.type)
        params = self.config.get('params', {})
        return layer_cls(input_dim=input_dim, **params)
