"""Transform builder for composing transforms from config."""

from src.builders.base import BaseBuilder
from src.registry import TRANSFORM_REGISTRY
from torchvision import transforms
import src.data.transforms  # noqa: F401


class TransformBuilder(BaseBuilder):
    """Builds composed transforms from configuration."""
    
    def build(self):
        """
        Build a composed transform from config.
        
        Config should be a list of dicts, each with:
        - name: transform name (registered in TRANSFORM_REGISTRY)
        - params: dict of parameters (optional)
        
        Returns:
            torchvision.transforms.Compose
        """
        if not self.config:
            # No transforms specified, return identity
            return transforms.Compose([])
        
        transform_list = []
        
        for transform_config in self.config:
            # Get transform name
            name = transform_config['name']
            
            # Get transform class from registry
            transform_cls = TRANSFORM_REGISTRY.get(name)
            
            # Get parameters (if any)
            params = transform_config.get('params', {})
            
            # Create transform instance
            transform = transform_cls(**params)
            
            transform_list.append(transform)
        
        return transforms.Compose(transform_list)

