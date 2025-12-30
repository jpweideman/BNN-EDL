"""CIFAR-10 dataset wrapper."""

from torchvision import datasets
from src.registry import DATASET_REGISTRY
from .base import BaseDataset


@DATASET_REGISTRY.register("cifar10")
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset with configurable transforms."""
    
    def __init__(self, config, data_dir, train_transform=None, test_transform=None):
        super().__init__(config, data_dir, train_transform, test_transform)
    
    def load_train_dataset(self):
        """Load CIFAR-10 training dataset."""
        return datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
    
    def load_test_dataset(self):
        """Load CIFAR-10 test dataset."""
        return datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )

