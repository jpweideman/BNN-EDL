"""Trainer creation — wires engines, handlers, and evaluators."""

from pathlib import Path
from src.optimizers.bnn import BNNOptimizer
from src.training.engine import create_train_engine
from src.training.bnn_engine import create_bnn_train_engine
from src.training.handlers import (
    attach_progress_bar_to_engine,
    attach_evaluator_handler,
    attach_wandb_logger_to_trainer,
    attach_wandb_logger_to_evaluator,
    attach_checkpoint_handler_to_evaluator,
    attach_last_checkpoint_handler,
    attach_early_stopping,
    attach_scheduler_handler,
    attach_annealing_handler
)


def create_trainer(model, optimizer, criterion, device, output_dir, evaluators,
                   scheduler=None, sampling_config=None, checkpoint_config=None,
                   wandb_config=None, early_stopping_config=None):
    """
    Create and configure training engine.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (standard or BNN)
        criterion: Loss function
        device: Device to run on
        output_dir: Output directory for checkpoints
        evaluators: Dict of evaluators from create_evaluators
        scheduler: Optional LR scheduler
        sampling_config: Optional BNN sampling configuration
        checkpoint_config: Optional checkpoint configuration
        wandb_config: Optional W&B configuration
        early_stopping_config: Optional early stopping configuration

    Returns:
        Configured training engine
    """
    is_bnn = isinstance(optimizer, BNNOptimizer)
    trainer = create_bnn_train_engine(model, optimizer, device) if is_bnn \
              else create_train_engine(model, optimizer, criterion, device)

    # Progress and W&B
    attach_progress_bar_to_engine(trainer, show_loss=True)
    if wandb_config is not None:
        attach_wandb_logger_to_trainer(trainer, wandb_config.log_interval)

    # BNN sampling
    sampling_manager = None
    if is_bnn and sampling_config is not None:
        from src.training.handlers.bnn import SamplingManager
        sampling_manager = SamplingManager(output_dir, sampling_config.start_epoch, sampling_config.sample_interval)
        sampling_manager.attach(trainer, model)
    elif is_bnn:
        print("Warning: BNN training without sampling. No ensemble evaluation possible.")
    trainer.sampling_manager = sampling_manager

    # Evaluators
    for split_name, eval_data in evaluators.items():
        evaluator = eval_data['evaluator']
        if sampling_manager:
            evaluator.sampling_manager = sampling_manager
        attach_evaluator_handler(trainer, evaluator, eval_data['loader'], split_name, eval_data['interval'])
        if wandb_config is not None:
            attach_wandb_logger_to_evaluator(evaluator, trainer, split_name, optimizer)

    # Epoch-level handlers
    if scheduler is not None:
        attach_scheduler_handler(trainer, scheduler)
    if hasattr(criterion, 'current_epoch'):
        attach_annealing_handler(trainer, criterion)

    # Early stopping
    early_stopping_handler = None
    if early_stopping_config is not None:
        early_stopping_handler = attach_early_stopping(
            evaluators[early_stopping_config.dataset]['evaluator'], trainer,
            early_stopping_config.metric, early_stopping_config.patience,
            early_stopping_config.min_delta, early_stopping_config.objective
        )
    trainer.early_stopping_handler = early_stopping_handler

    # Checkpointing
    if checkpoint_config is not None:
        checkpoint_split = checkpoint_config.dataset
        if checkpoint_split not in evaluators:
            print(f"Warning: Checkpoint configured for '{checkpoint_split}' but no evaluator found. Checkpointing disabled.")
        else:
            attach_checkpoint_handler_to_evaluator(
                evaluators[checkpoint_split]['evaluator'], model, trainer, optimizer,
                checkpoint_config.metric, checkpoint_config.objective,
                Path(output_dir) / "best_model.pt",
                scheduler=scheduler, early_stopping_handler=early_stopping_handler
            )
            attach_last_checkpoint_handler(
                trainer, model, optimizer, Path(output_dir) / "last_checkpoint.pt",
                scheduler, sampling_manager=sampling_manager,
                early_stopping_handler=early_stopping_handler
            )

    return trainer
