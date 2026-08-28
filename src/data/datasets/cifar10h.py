"""CIFAR-10H dataset wrapper: CIFAR-10 test images paired with human count vectors."""

import urllib.request
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets
from src.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("cifar10h")
class CIFAR10HDataset:
    """CIFAR-10H dataset loader."""

    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory to store/load data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_source(self, source_name):
        """
        Load the paired CIFAR-10H source dataset.

        Args:
            source_name: 'all' (the complete 10,000-image paired dataset)

        Returns:
            CIFAR10HCounts
        """
        if source_name != 'all':
            raise ValueError(f"CIFAR10H only supports the 'all' source, got '{source_name}'")
        return CIFAR10HCounts(self.data_dir)


class CIFAR10HCounts:
    """CIFAR-10 test images paired with their CIFAR-10H human count vectors.

    One item is (image, counts) with counts as a float32 vector of length 10.
    The original CIFAR-10 class labels stay available through `targets`.
    """

    def __init__(self, data_dir):
        self.images = datasets.CIFAR10(root=data_dir, train=False, download=True)
        self.targets = self.images.targets
        counts_path = Path(data_dir) / "cifar10h" / "cifar10h-counts.npy"
        if not counts_path.exists():
            url = ("https://raw.githubusercontent.com/jcpeterson/cifar-10h/"
                   "master/data/cifar10h-counts.npy")
            counts_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, counts_path)
        self.counts = torch.as_tensor(np.load(counts_path), dtype=torch.float32)

    def __getitem__(self, idx):
        image, _ = self.images[idx]
        return image, self.counts[idx]

    def __len__(self):
        return len(self.images)
