"""CIFAR-10 dataset wrapper."""

from torchvision import datasets
from src.registry import DATASET_REGISTRY
from .base import BaseDataset


@DATASET_REGISTRY.register("cifar10")
class CIFAR10Dataset(BaseDataset):
    """
    CIFAR-10 dataset.

        source: train  # 50,000 images
        source: test   # 10,000 images
    """
    
    def load_source_dataset(self, source_name):
        """
        Load CIFAR-10 source dataset without transforms.
        
        Args:
            source_name: 'train' or 'test'
        
        Returns:
            torchvision.datasets.CIFAR10 
        """
        if source_name == 'train':
            return datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
            )
        elif source_name == 'test':
            return datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
            )
        else:
            raise ValueError(f"Cifar10 only supports 'train' or 'test' sources, got '{source_name}'")
