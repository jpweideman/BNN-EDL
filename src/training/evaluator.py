"""Evaluator creation with metrics."""

from src.optimizers.bnn import BNNOptimizer
from src.builders.metrics_builder import MetricsBuilder
from src.training.engine import create_eval_engine
from src.training.handlers import attach_progress_bar_to_engine


def create_evaluators(model, criterion, device, loaders, evaluation_config, optimizer=None):
    """
    Create evaluation engines for all configured splits.

    Args:
        model: PyTorch model
        criterion: Loss function
        device: Device to run on
        loaders: Dict mapping split names to DataLoaders
        evaluation_config: Evaluation configuration for all splits
        optimizer: Optimizer (used to detect if BNN training)

    Returns:
        Dict mapping split names to {evaluator, loader, interval}
    """
    is_bnn = isinstance(optimizer, BNNOptimizer)
    evaluators = {}
    for split_name, eval_config in evaluation_config.items():
        if split_name not in loaders:
            print(f"Warning: '{split_name}' not in loaders. Skipping.")
            continue

        if is_bnn:
            from src.training.bnn_engine import create_bnn_eval_engine
            evaluator = create_bnn_eval_engine(model, criterion, device)
        else:
            evaluator = create_eval_engine(model, criterion, device)

        metrics = MetricsBuilder(eval_config.metrics).build()
        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        attach_progress_bar_to_engine(evaluator, show_loss=False)
        evaluators[split_name] = {
            'evaluator': evaluator,
            'loader': loaders[split_name],
            'interval': eval_config.interval
        }

    return evaluators
