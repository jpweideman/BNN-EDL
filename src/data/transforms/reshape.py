"""Reshape and conversion transforms."""

from src.registry import TRANSFORM_REGISTRY
from torchvision import transforms


@TRANSFORM_REGISTRY.register("to_tensor")
class ToTensor:
    """Convert PIL Image or numpy array to tensor."""
    
    def __init__(self):
        self.transform = transforms.ToTensor()
    
    def __call__(self, x):
        return self.transform(x)


@TRANSFORM_REGISTRY.register("flatten")
class Flatten:
    """Flatten image to 1D vector (for MLP)."""
    
    def __init__(self):
        self.transform = transforms.Lambda(lambda x: x.view(-1))
    
    def __call__(self, x):
        return self.transform(x)
