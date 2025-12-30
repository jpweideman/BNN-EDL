"""Data augmentation transforms."""

from src.registry import TRANSFORM_REGISTRY
from torchvision import transforms


@TRANSFORM_REGISTRY.register("random_crop")
class RandomCrop:
    """Random crop with padding."""
    
    def __init__(self, size, padding):
        """
        Args:
            size: Output size
            padding: Padding size
        """
        self.transform = transforms.RandomCrop(size, padding=padding)
    
    def __call__(self, x):
        return self.transform(x)


@TRANSFORM_REGISTRY.register("random_horizontal_flip")
class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, p):
        """
        Args:
            p: Probability of flip
        """
        self.transform = transforms.RandomHorizontalFlip(p=p)
    
    def __call__(self, x):
        return self.transform(x)


