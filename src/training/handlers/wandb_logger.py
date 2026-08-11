"""W&B logging handlers.

Metric names are '<section>/<metric>': W&B groups its panels by the part
before the slash, giving training and each evaluation split its own section.
"""

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
            **{f'train/{k}': v for k, v in output.items()},
            'iteration': engine.state.iteration
        })


def attach_wandb_logger_to_evaluator(evaluator, trainer, prefix, optimizer):
    """
    Attach W&B logging to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        trainer: Training engine (for epoch number)
        prefix: Section for metric names, i.e. the split (e.g. 'cifar10_test')
        optimizer: Optimizer (for learning rate)
    """
    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics(engine):
        # Get learning rate (handle both standard and BNN optimizers)
        lr = (optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups')
              else optimizer.lr)

        wandb.log({
            **{f'{prefix}/{k}': v for k, v in engine.state.metrics.items()},
            'epoch': trainer.state.epoch,
            'train/learning_rate': lr
        })

