"""Base dataset class."""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for all datasets."""
    
    def __init__(self, config, data_dir, seed):
        """
        Args:
            config: Dataset configuration object
            data_dir: Directory to store/load data
            seed: Random seed for reproducible splits
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.seed = seed
    
    @abstractmethod
    def load_source_dataset(self, source_name):
        """
        Load a source dataset without transforms.
        
        Args:
            source_name: Name of the source (dataset-specific)
        
        Returns:
            torch.utils.data.Dataset 
        """
        pass
    
    def get_loaders(self, transforms_dict):
        """
        Build and return data loaders based on config splits.
        
        Args:
            transforms_dict: Dict mapping split names to transforms
        
        Returns:
            dict: Dictionary mapping split names to DataLoaders
        """
        # Group splits by their source (e.g. train or test)
        splits_by_source = {}
        for split_name, split_config in self.config.splits.items():
            source = split_config.source
            if source not in splits_by_source:
                splits_by_source[source] = []
            splits_by_source[source].append((split_name, split_config))
        
        # Validate: fractions for each source must sum to 1
        for source, splits in splits_by_source.items():
            total = sum(cfg.fraction for _, cfg in splits)
            if total != 1.0:
                raise ValueError(f"Fractions for source '{source}' sum to {total:.4f}, must equal 1.0")
        
        loaders = {}
        
        # For each source: load data, split according to fractions, create loaders
        for source, splits in splits_by_source.items():
            # Load raw data from this source
            raw_data = self.load_source_dataset(source)
            
            # Get fractions for splitting
            fractions = [cfg.fraction for _, cfg in splits]
            
            # Split data into non-overlapping subsets
            subsets = random_split(raw_data, fractions, generator=torch.Generator().manual_seed(self.seed))
            
            # Create a loader for each split
            for (split_name, split_config), subset in zip(splits, subsets):
                # Apply transform to this split
                transform = transforms_dict.get(split_name)
                if transform:
                    subset = _TransformedDataset(subset, transform)
                
                # Create DataLoader
                loaders[split_name] = DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=split_config.get('shuffle', False),
                    num_workers=self.num_workers,
                    pin_memory=True
                )
        
        return loaders


class _TransformedDataset:
    """Helper to apply transform to a dataset or subset."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.dataset)
