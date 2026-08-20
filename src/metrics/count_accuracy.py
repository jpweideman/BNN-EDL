"""Accuracy against the majority count class."""

from ignite.metrics import Accuracy as IgniteAccuracy
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("count_accuracy")
class CountAccuracy:
    """
    Accuracy of the predicted class against the majority class of the counts.
    """

    def __init__(self):
        """Initialize count accuracy metric."""
        self.metric = IgniteAccuracy(
            output_transform=lambda output: (output['y_pred'], output['y'].argmax(dim=-1)))

    def attach(self, engine, name):
        """Attach metric to an Ignite engine."""
        self.metric.attach(engine, name)
