"""Evaluator builder for creating evaluation engines."""

from src.builders.base import BaseBuilder
from src.builders.metrics_builder import MetricsBuilder
from src.training.engine import create_eval_engine


class EvaluatorBuilder(BaseBuilder):
    """Builds evaluation engines with metrics."""
    
    def build(self, model, criterion, device, sample_files=None):
        """
        Build an evaluation engine with metrics.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Device to run on
            sample_files: Optional list of sample files for BNN evaluation
        
        Returns:
            Evaluation engine with metrics attached
        """
        # Create appropriate engine based on whether sample_files provided
        if sample_files is not None:
            from src.training.bnn_engine import create_bnn_eval_engine
            evaluator = create_bnn_eval_engine(model, criterion, device, sample_files)
        else:
            evaluator = create_eval_engine(model, criterion, device)
        
        # Build and attach metrics 
        metrics = MetricsBuilder(self.config).build()
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        
        return evaluator

