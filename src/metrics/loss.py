"""Loss metric wrapper."""

from ignite.metrics import Average
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("loss")
class Loss:
    """
    Loss metric that averages loss values from eval_step.
    """
    
    def __init__(self):
        """Initialize loss metric."""
        self.metric = Average(output_transform=lambda output: output['loss'].cpu())
    
    def attach(self, engine, name):
        """Attach metric to an Ignite engine."""
        self.metric.attach(engine, name)

