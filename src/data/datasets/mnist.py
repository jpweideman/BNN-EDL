"""MNIST dataset wrapper."""

from pathlib import Path
from torchvision import datasets
from src.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("mnist")
class MNISTDataset:
    """MNIST dataset loader."""
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory to store/load data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_source(self, source_name):
        """
        Load MNIST source dataset.
        
        Args:
            source_name: 'train' or 'test'
        
        Returns:
            torchvision.datasets.MNIST 
        """
        if source_name == 'train':
            return datasets.MNIST(root=self.data_dir, train=True, download=True)
        elif source_name == 'test':
            return datasets.MNIST(root=self.data_dir, train=False, download=True)
        else:
            raise ValueError(f"MNIST only supports 'train' or 'test' sources, got '{source_name}'")
