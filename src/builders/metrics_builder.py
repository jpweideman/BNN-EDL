"""Metrics builder."""

from src.builders.base import BaseBuilder
from src.registry import METRIC_REGISTRY
import src.metrics  # noqa: F401 # Triggers registration


class MetricsBuilder(BaseBuilder):
    """Builds metrics from configuration."""
    
    def build(self):
        """
        Build metrics from configuration.
        
        Returns:
            Dictionary of metric instances {name: metric}
        """
        metrics = {}
        
        for metric_config in self.config:
            name = metric_config['name']
            metric_cls = METRIC_REGISTRY.get(name)
            
            # Get parameters if specified in config
            params = metric_config.get('params', {})
            metric = metric_cls(**params)
            metrics[name] = metric
        
        return metrics

