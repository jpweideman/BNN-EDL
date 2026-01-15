"""Early stopping handler."""

from ignite.engine import Events
from ignite.handlers import EarlyStopping
from src.utils.objective import get_score_sign


def attach_early_stopping(evaluator, trainer, metric_name, patience, min_delta, objective):
    """
    Attach early stopping to an evaluator.
    
    Args:
        evaluator: Evaluation engine to monitor
        trainer: Training engine to stop
        metric_name: Metric to track (e.g. 'loss' or 'accuracy')
        patience: Epochs without improvement before stopping
        min_delta: Minimum change to count as improvement
        objective: 'minimize' for lower is better (e.g. loss), 'maximize' for higher is better (e.g. accuracy)
    
    Returns:
        EarlyStopping handler instance
    """    
    def score_function(engine):
        return get_score_sign(objective) * engine.state.metrics[metric_name]
    
    handler = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer,
        min_delta=min_delta
    )
    
    evaluator.add_event_handler(Events.COMPLETED, handler)
    return handler