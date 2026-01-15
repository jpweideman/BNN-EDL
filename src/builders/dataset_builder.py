"""Dataset builder for instantiating datasets from registry."""

from src.builders.base import BaseBuilder
from src.registry import DATASET_REGISTRY
import src.data.datasets  # noqa: F401


class DatasetBuilder(BaseBuilder):
    """Builds dataset instances from configuration."""
    
    def build(self, data_dir):
        """
        Build a dataset from configuration.
        
        Args:
            data_dir: Directory containing data
        
        Returns:
            Dataset instance
        """
        dataset_cls = DATASET_REGISTRY.get(self.config.name)
        return dataset_cls(data_dir)

