"""SVHN dataset wrapper."""

from pathlib import Path
from torchvision import datasets
from src.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("svhn")
class SVHNDataset:
    """SVHN dataset loader, used as the CIFAR-10 OOD set.

    Needs scipy, which torchvision uses to read SVHN's .mat files.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory to store/load data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_source(self, source_name):
        """
        Load SVHN source dataset.

        Args:
            source_name: 'train' or 'test'

        Returns:
            torchvision.datasets.SVHN
        """
        if source_name == 'train':
            return datasets.SVHN(root=self.data_dir, split='train', download=True)
        elif source_name == 'test':
            return datasets.SVHN(root=self.data_dir, split='test', download=True)
        else:
            raise ValueError(f"SVHN only supports 'train' or 'test' sources, got '{source_name}'")
