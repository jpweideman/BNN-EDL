"""Data loader builder."""

from src.builders.base import BaseBuilder
from src.builders.transform_builder import TransformBuilder
from src.registry import DATASET_REGISTRY
from pathlib import Path
import src.data.datasets  # noqa: F401


class DataLoaderBuilder(BaseBuilder):
    """Builds data loaders with configurable transforms and splits."""
    
    def build(self, seed):
        """
        Build data loaders based on config splits.
        
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
            else:
                print(f"Warning: No transforms specified for split '{split_name}'")
                transforms_dict[split_name] = None
        
        # Get dataset class from registry
        dataset_cls = DATASET_REGISTRY.get(self.config.name)
        
        # Create dataset instance
        dataset = dataset_cls(
            config=self.config,
            data_dir=str(data_dir),
            seed=seed
        )
        
        # Get loaders (returns dict)
        return dataset.get_loaders(transforms_dict)
