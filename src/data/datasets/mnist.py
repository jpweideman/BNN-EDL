"""MNIST dataset wrapper."""

from torchvision import datasets
from src.registry import DATASET_REGISTRY
from .base import BaseDataset


@DATASET_REGISTRY.register("mnist")
class MNISTDataset(BaseDataset):
    """
    MNIST dataset.

        source: train  # 60,000 images
        source: test   # 10,000 images
    """
    
    def load_source_dataset(self, source_name):
        """
        Load MNIST source dataset without transforms.
        
        Args:
            source_name: 'train' or 'test'
        
        Returns:
            torchvision.datasets.MNIST 
        """
        if source_name == 'train':
            return datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
            )
        elif source_name == 'test':
            return datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
            )
        else:
            raise ValueError(f"MNIST only supports 'train' or 'test' sources, got '{source_name}'")
