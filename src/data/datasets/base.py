"""Base dataset class."""

import torch
import warnings
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for all datasets."""
    
    def __init__(
        self,
        config,
        data_dir,
        train_transform=None,
        test_transform=None
    ):
        """
        Args:
            config: Dataset configuration object
            data_dir: Directory to store/load data
            train_transform: Transform for training data
            test_transform: Transform for test/val data
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_split = getattr(config, 'train_split', 0.9)
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
    
    @abstractmethod
    def load_train_dataset(self):
        """Load the training dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_test_dataset(self):
        """Load the test dataset. Must be implemented by subclasses."""
        pass
    
    def get_loaders(self):
        """
        Build and return data loaders.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        if self._train_loader is not None:
            return self._train_loader, self._val_loader, self._test_loader
        
        # Load datasets (implemented by subclass)
        train_dataset = self.load_train_dataset()
        test_dataset = self.load_test_dataset()
        
        # Split training into train/val
        if self.train_split < 1.0:
            train_size = int(self.train_split * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            warnings.warn(
                "train_split=1.0: Using test set as validation set. "
                "Consider using train_split < 1.0 for proper train/val/test splits.",
                UserWarning
            )
            val_dataset = test_dataset
        
        # Create data loaders
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self._val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self._test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return self._train_loader, self._val_loader, self._test_loader

