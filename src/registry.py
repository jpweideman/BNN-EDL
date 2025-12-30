"""
Central registry system for pluggable components.

This enables a plugin architecture where new models, losses, metrics, etc.
can be added by decorating them with @REGISTRY.register("name").
"""

class Registry:
    """Registry for registering and retrieving components by name."""
    
    def __init__(self, name: str):
        """
        Initialize a registry.
        
        Args:
            name: Name of the registry (for error messages)
        """
        self.name = name
        self._registry = {}
    
    def register(self, name: str):
        """
        Decorator to register a component.
        
        Usage example:
            @MODEL_REGISTRY.register("mlp")
            class MLP(nn.Module):
                ...
        
        Args:
            name: Name to register the component under
        """
        def decorator(component):
            if name in self._registry:
                raise ValueError(
                    f"'{name}' is already registered in {self.name} registry. "
                    f"Cannot register {component}."
                )
            self._registry[name] = component
            return component
        return decorator
    
    def get(self, name: str):
        """
        Get a registered component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            The registered component
            
        Raises:
            KeyError: If component not found
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]


# Create global registries for each component type
MODEL_REGISTRY = Registry("models")
LOSS_REGISTRY = Registry("losses")
METRIC_REGISTRY = Registry("metrics")
SAMPLER_REGISTRY = Registry("samplers")
DATASET_REGISTRY = Registry("datasets")
TRANSFORM_REGISTRY = Registry("transforms")

