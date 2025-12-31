"""Accuracy metric for classification."""

from ignite.metrics import Accuracy as IgniteAccuracy
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("accuracy")
class Accuracy:
    """
    Accuracy metric using Ignite's implementation.
    """
    
    def __init__(self):
        """Initialize accuracy metric."""
        self.metric = IgniteAccuracy(output_transform=lambda output: (output['y_pred'], output['y']))
    
    def attach(self, engine, name):
        """Attach metric to an Ignite engine."""
        self.metric.attach(engine, name)

