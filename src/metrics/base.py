"""Base metric class for custom metrics."""

from ignite.metrics import Metric


class BaseMetric(Metric):
    """Base class for metrics that override iteration_completed."""
    
    def update(self, output):
        """Required by Ignite but unused since we override iteration_completed."""
        pass
