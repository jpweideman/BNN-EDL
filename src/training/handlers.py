"""Generic event handlers for training."""

import torch
from pathlib import Path
from ignite.engine import Events
from ignite.contrib.handlers import ProgressBar


def attach_progress_bar_to_engine(engine, show_loss=True):
    """
    Attach progress bar to any engine.
    
    Args:
        engine: Ignite engine
        show_loss: Whether to show loss in progress bar
    """
    pbar = ProgressBar()
    if show_loss:
        pbar.attach(engine, output_transform=lambda output: {'loss': output})
    else:
        pbar.attach(engine)


def attach_evaluator_handler(trainer, evaluator, loader, name, interval):
    """
    Attach evaluation handler for any dataset.
    
    Args:
        trainer: Training engine
        evaluator: Evaluation engine
        loader: Data loader
        name: Display name for this evaluation
        interval: Evaluation frequency (0=disabled, -1=last epoch only, >0=every N epochs)
    """
    if interval == 0:
        return
    
    if interval == -1:
        @trainer.on(Events.COMPLETED)
        def run_final_eval(engine):
            evaluator.run(loader)
            print(f"\nFinal {name} Evaluation:")
            for k, v in evaluator.state.metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        @trainer.on(Events.EPOCH_COMPLETED(every=interval))
        def run_periodic_eval(engine):
            evaluator.run(loader)
            print(f"\nEpoch {engine.state.epoch} - {name}:")
            for k, v in evaluator.state.metrics.items():
                print(f"  {k}: {v:.4f}")


def attach_wandb_logger_to_trainer(trainer, log_interval):
    """
    Attach W&B logging for training loss.
    
    Args:
        trainer: Training engine
        log_interval: Log every N iterations
    """
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B logging.")
        return
    
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_train_loss(engine):
        wandb.log({
            'train_loss': engine.state.output,
            'iteration': engine.state.iteration
        })


def attach_wandb_logger_to_evaluator(evaluator, trainer, prefix):
    """
    Attach W&B logging to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        trainer: Training engine (for epoch number)
        prefix: Prefix for metric names (e.g., 'val', 'test')
    """
    try:
        import wandb
    except ImportError:
        return
    
    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics(engine):
        wandb.log({
            **{f'{prefix}_{k}': v for k, v in engine.state.metrics.items()},
            'epoch': trainer.state.epoch
        })


def attach_checkpoint_handler_to_evaluator(evaluator, model, metric_name, filepath):
    """
    Attach checkpointing to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        model: Model to save
        metric_name: Metric to track for best model
        filepath: Path to save checkpoint
    """
    best_metric = -float('inf')
    
    @evaluator.on(Events.COMPLETED)
    def save_best_checkpoint(engine):
        nonlocal best_metric
        
        if metric_name not in engine.state.metrics:
            print(f" Metric '{metric_name}' not found. Available: {list(engine.state.metrics.keys())}")
            return
        
        current = engine.state.metrics[metric_name]
        if current > best_metric:
            best_metric = current
            torch.save(model.state_dict(), filepath)
            print(f"Saved best model ({metric_name}={current:.4f}) to {filepath}")
