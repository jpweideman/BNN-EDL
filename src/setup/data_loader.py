"""DataLoader orchestrator for creating loaders with transforms."""

import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from src.builders.dataset_builder import DatasetBuilder
from src.builders.transform_builder import TransformBuilder


class DataLoaderSetup:
    """Orchestrates dataset creation and loader setup with transforms."""
    
    def __init__(self, dataset_config):
        """
        Initialize with dataset configuration.
        
        Args:
            dataset_config: Dataset configuration (splits, transforms, etc.)
        """
        self.config = dataset_config
    
    def create_loaders(self, seed):
        """
        Create data loaders for all configured splits.
        
        Args:
            seed: Random seed for reproducible data splitting
        
        Returns:
            dict: Dictionary mapping split names to DataLoaders
        """
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        
        # Build transforms for each split
        transforms_dict = {}
        for split_name, split_config in self.config.splits.items():
            if hasattr(split_config, 'transforms') and split_config.transforms:
                transforms_dict[split_name] = TransformBuilder(split_config.transforms).build()
        
        # Build dataset 
        dataset = DatasetBuilder(self.config).build(str(data_dir))
        
        # Group splits by their source (e.g. train, val, test)
        splits_by_source = {}
        for split_name, split_config in self.config.splits.items():
            source = split_config.source
            if source not in splits_by_source:
                splits_by_source[source] = []
            splits_by_source[source].append((split_name, split_config))
        
        # Validate: fractions for each source must sum to 1
        for source, splits in splits_by_source.items():
            total = sum(split_cfg.fraction for _, split_cfg in splits)
            if total != 1.0:
                raise ValueError(f"Fractions for source '{source}' sum to {total:.4f}, must equal 1.0")
        
        # Create DataLoaders
        loaders = {}
        for source, splits in splits_by_source.items():
            # Load raw data from this source
            raw_data = dataset.load_source(source)
            
            # Get fractions for splitting
            fractions = [split_cfg.fraction for _, split_cfg in splits]
            
            # Split data into non-overlapping subsets
            subsets = random_split(raw_data, fractions, generator=torch.Generator().manual_seed(seed))
            
            # Create a loader for each split
            for (split_name, split_config), subset in zip(splits, subsets):
                # Apply transform to this split
                transform = transforms_dict.get(split_name)
                if transform:
                    subset = _TransformedDataset(subset, transform)
                
                # Create DataLoader
                loaders[split_name] = DataLoader(
                    subset,
                    batch_size=self.config.batch_size,
                    shuffle=split_config.get('shuffle', False),
                    num_workers=self.config.num_workers,
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
