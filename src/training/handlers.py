"""Event handlers for training."""

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


def attach_wandb_logger_to_evaluator(evaluator, trainer, prefix, optimizer):
    """
    Attach W&B logging to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        trainer: Training engine (for epoch number)
        prefix: Prefix for metric names (e.g., 'val', 'test')
        optimizer: Optimizer (for learning rate)
    """
    try:
        import wandb
    except ImportError:
        return
    
    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics(engine):
        wandb.log({
            **{f'{prefix}_{k}': v for k, v in engine.state.metrics.items()},
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': trainer.state.epoch
        })


def attach_checkpoint_handler_to_evaluator(evaluator, model, trainer, optimizer, metric_name, filepath):
    """
    Attach checkpointing to any evaluator.
    
    Args:
        evaluator: Evaluation engine
        model: Model to save
        trainer: Training engine (for epoch number)
        optimizer: Optimizer to save
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
            try:
                import wandb
                wandb_run_id = wandb.run.id if wandb.run else None
            except (ImportError, AttributeError):
                wandb_run_id = None
            
            checkpoint = {
                'epoch': trainer.state.epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metric_name': metric_name,
                'wandb_run_id': wandb_run_id
            }
            torch.save(checkpoint, filepath)

def attach_last_checkpoint_handler(trainer, model, optimizer, filepath, scheduler=None):
    """
    Save checkpoint after every epoch to enable resuming.
    
    Args:
        trainer: Training engine
        model: Model to save
        optimizer: Optimizer to save
        filepath: Path to save checkpoint
        scheduler: Optional scheduler to save
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_last_checkpoint(engine):
        try:
            import wandb
            wandb_run_id = wandb.run.id if wandb.run else None
        except (ImportError, AttributeError):
            wandb_run_id = None
        
        checkpoint = {
            'epoch': engine.state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_run_id': wandb_run_id
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filepath)


def attach_early_stopping(evaluator, trainer, metric_name, patience, min_delta, mode):
    """
    Attach early stopping to an evaluator.
    
    Args:
        evaluator: Evaluation engine to monitor
        trainer: Training engine to stop
        metric_name: Metric to track ( eg. 'loss' or 'accuracy')
        patience: Epochs without improvement before stopping
        min_delta: Minimum change to count as improvement
        mode: 'minimize' for lower is better (eg. loss), 'maximize' for higher is better (eg. accuracy)
    """
    from ignite.handlers import EarlyStopping
    
    def score_function(engine):
        metric = engine.state.metrics[metric_name]
        if mode == 'minimize':
            return -metric
        elif mode == 'maximize':
            return metric
    
    handler = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer,
        min_delta=min_delta
    )
    
    evaluator.add_event_handler(Events.COMPLETED, handler)


def attach_scheduler_handler(trainer, scheduler):
    """
    Attach scheduler to update learning rate after each epoch.
    
    Args:
        trainer: Training engine
        scheduler: PyTorch scheduler instance
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_scheduler(engine):
        scheduler.step()
