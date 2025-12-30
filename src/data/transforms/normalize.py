"""Normalization transforms."""

from src.registry import TRANSFORM_REGISTRY
from torchvision import transforms


@TRANSFORM_REGISTRY.register("normalize")
class Normalize:
    """Normalize images with configurable mean and std."""
    
    def __init__(self, mean, std):
        """
        Args:
            mean: Mean values (list or tuple)
            std: Std values (list or tuple)
        """
        self.transform = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, x):
        return self.transform(x)

