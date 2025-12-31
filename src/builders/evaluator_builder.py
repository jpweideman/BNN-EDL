"""Evaluator builder for creating evaluation engines."""

from src.builders.base import BaseBuilder
from src.builders.metrics_builder import MetricsBuilder
from src.training.engine import create_eval_engine


class EvaluatorBuilder(BaseBuilder):
    """Builds evaluation engines with metrics."""
    
    def build(self, model, criterion, device):
        """
        Build an evaluation engine with metrics.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Device to run on
        
        Returns:
            Evaluation engine with metrics attached
        """
        evaluator = create_eval_engine(model, criterion, device)
        
        # Build and attach metrics
        metrics = MetricsBuilder(self.config).build()
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        
        return evaluator

