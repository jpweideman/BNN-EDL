"""Data loader builder."""

from src.builders.base import BaseBuilder
from src.builders.transform_builder import TransformBuilder
from src.registry import DATASET_REGISTRY
from pathlib import Path
import src.data.datasets  # noqa: F401


class DataLoaderBuilder(BaseBuilder):
    """Builds data loaders with configurable transforms."""
    
    def build(self):
        """
        Build train, val, test loaders.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Get project root for data directory
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        
        # Build transforms
        train_transform = None
        test_transform = None
        
        if hasattr(self.config, 'train_transforms') and self.config.train_transforms:
            train_transform = TransformBuilder(self.config.train_transforms).build()
        
        if hasattr(self.config, 'test_transforms') and self.config.test_transforms:
            test_transform = TransformBuilder(self.config.test_transforms).build()
        
        # Get dataset class from registry
        dataset_cls = DATASET_REGISTRY.get(self.config.name)
        
        # Create dataset instance with config
        dataset = dataset_cls(
            config=self.config,
            data_dir=str(data_dir),
            train_transform=train_transform,
            test_transform=test_transform
        )
        
        # Get loaders from dataset
        return dataset.get_loaders()

