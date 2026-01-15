"""W&B logging handlers."""

import wandb
from ignite.engine import Events


def attach_wandb_logger_to_trainer(trainer, log_interval):
    """
    Attach W&B logging for training loss.
    
    Args:
        trainer: Training engine
        log_interval: Log every N iterations
    """
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_train_metrics(engine):
        output = engine.state.output
        
        wandb.log({
            **{f'train_{k}': v for k, v in output.items()},
            'iteration': engine.state.iteration
        })


def attach_wandb_logger_to_evaluator(evaluator, trainer, prefix, optimizer):
    """
    Attach W&B logging to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        trainer: Training engine (for epoch number)
        prefix: Prefix for metric names (e.g., 'val', 'test')
        optimizer: Optimizer (for learning rate)
    """
    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics(engine):
        wandb.log({
            **{f'{prefix}_{k}': v for k, v in engine.state.metrics.items()},
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': trainer.state.epoch
        })

