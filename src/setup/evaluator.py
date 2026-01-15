"""Evaluator setup for creating evaluation engines with metrics."""

from src.builders.metrics_builder import MetricsBuilder
from src.training.engine import create_eval_engine
from src.training.handlers import attach_progress_bar_to_engine


class EvaluatorSetup:
    """Orchestrates evaluator creation with metrics."""
    
    def __init__(self, evaluation_config):
        """
        Initialize with evaluation configuration.
        
        Args:
            evaluation_config: Evaluation configuration for all splits
        """
        self.evaluation_config = evaluation_config
    
    def create_evaluators(self, model, criterion, device, loaders, sample_files=None):
        """
        Create evaluation engines for all configured splits.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            device: Device to run on
            loaders: Dict mapping split names to DataLoaders
            sample_files: Optional list of sample files for BNN evaluation
        
        Returns:
            Dict mapping split names to configured evaluators
        """
        evaluators = {}
        for split_name, eval_config in self.evaluation_config.items():
            if split_name not in loaders:
                print(f"Warning: '{split_name}' not in loaders. Skipping.")
                continue
            
            evaluator = self._create_evaluator(model, criterion, device, eval_config.metrics, sample_files)
            evaluators[split_name] = {
                'evaluator': evaluator,
                'loader': loaders[split_name],
                'interval': eval_config.interval
            }
            
            attach_progress_bar_to_engine(evaluator, show_loss=False)
        
        return evaluators
    
    def _create_evaluator(self, model, criterion, device, metrics_config, sample_files=None):
        """Create a single evaluator with metrics."""
        if sample_files is not None:
            from src.training.bnn_engine import create_bnn_eval_engine
            evaluator = create_bnn_eval_engine(model, criterion, device, sample_files)
        else:
            evaluator = create_eval_engine(model, criterion, device)
        
        # Build and attach metrics 
        metrics = MetricsBuilder(metrics_config).build()
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        
        return evaluator
