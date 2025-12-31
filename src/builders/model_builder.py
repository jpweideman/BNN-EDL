"""Model builder for instantiating models from config."""

from src.builders.base import BaseBuilder
from src.registry import MODEL_REGISTRY
import src.models  # noqa: F401 # Triggers registration of models


class ModelBuilder(BaseBuilder):
    """Builds model instances from configuration."""
    
    def build(self):
        """
        Build a model from configuration.
        
        Returns:
            torch.nn.Module: Instantiated model
        """
        model_cls = MODEL_REGISTRY.get(self.config.name)
        params = self.config.params
        model = model_cls(**params)
        return model

