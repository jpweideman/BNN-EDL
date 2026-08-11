"""Checkpoint handlers."""

import torch
import wandb
from pathlib import Path
from ignite.engine import Events
from src.utils.objective import get_comparison_fn


def attach_checkpoint_handler_to_evaluator(evaluator, model, trainer, optimizer, metric_name, objective, filepath, scheduler=None, early_stopping_handler=None):
    """
    Attach checkpointing to an evaluator.
    
    Args:
        evaluator: Evaluation engine
        model: Model to save
        trainer: Training engine (for epoch number)
        optimizer: Optimizer to save
        metric_name: Metric to track for best model
        objective: 'minimize' or 'maximize' the metric
        filepath: Path to save checkpoint
        scheduler: Optional scheduler to save
        early_stopping_handler: Optional early stopping handler to save state
    """
    best_metric, is_better = get_comparison_fn(objective)
    
    @evaluator.on(Events.COMPLETED)
    def save_best_checkpoint(engine):
        nonlocal best_metric
        
        if metric_name not in engine.state.metrics:
            print(f" Metric '{metric_name}' not found. Available: {list(engine.state.metrics.keys())}")
            return
        
        current = engine.state.metrics[metric_name]
        if is_better(current, best_metric):
            best_metric = current
            wandb_run_id = wandb.run.id if wandb.run else None
            
            checkpoint = {
                'epoch': trainer.state.epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'metric_name': metric_name,
                'wandb_run_id': wandb_run_id,
                'rng_state': {
                    'torch': torch.get_rng_state().cpu().to(torch.uint8),
                    'torch_cuda': [state.cpu().to(torch.uint8) for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
                }
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            if hasattr(trainer, 'sampling_manager') and trainer.sampling_manager is not None:
                checkpoint['sample_files'] = [str(p) for p in trainer.sampling_manager.get_sample_files()]
            
            if early_stopping_handler is not None:
                checkpoint['early_stopping_state'] = {
                    'counter': early_stopping_handler.counter,
                    'best_score': early_stopping_handler.best_score,
                }
            
            torch.save(checkpoint, filepath)


def attach_best_checkpoint_restore(trainer, model, filepath, device):
    """
    Restore the best checkpoint's weights when training completes.

    Attached before the evaluator handlers, so the final (interval=-1)
    evaluations report the selected model instead of the last epoch. Only for
    deterministic runs: sampled runs report the BMA over their snapshots.

    Args:
        trainer: Training engine
        model: Model to restore the weights into
        filepath: Path to the best checkpoint
        device: Device to map the checkpoint to
    """
    @trainer.on(Events.COMPLETED)
    def restore_best_checkpoint(engine):
        if not Path(filepath).exists():
            print(f"No checkpoint at {filepath}. Final evaluation uses the last epoch.")
            return

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nRestored best checkpoint for the final evaluation: epoch "
              f"{checkpoint.get('epoch')}, {checkpoint.get('metric_name')}="
              f"{checkpoint.get('best_metric')}")


def attach_last_checkpoint_handler(trainer, model, optimizer, filepath, scheduler=None, sampling_manager=None, early_stopping_handler=None):
    """
    Save checkpoint after every epoch to enable resuming.
    
    Args:
        trainer: Training engine
        model: Model to save
        optimizer: Optimizer to save
        filepath: Path to save checkpoint
        scheduler: Optional scheduler to save
        sampling_manager: Optional sampling manager for BNN training
        early_stopping_handler: Optional early stopping handler to save state
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_last_checkpoint(engine):
        wandb_run_id = wandb.run.id if wandb.run else None
        
        checkpoint = {
            'epoch': engine.state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_run_id': wandb_run_id,
            'rng_state': {
                'torch': torch.get_rng_state().cpu().to(torch.uint8),
                'torch_cuda': [state.cpu().to(torch.uint8) for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
            }
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save sample files for BNN. Convert Path to str for torch.load compatibility
        if sampling_manager is not None:
            checkpoint['sample_files'] = [str(p) for p in sampling_manager.get_sample_files()]
        
        # Save early stopping state
        if early_stopping_handler is not None:
            checkpoint['early_stopping_state'] = {
                'counter': early_stopping_handler.counter,
                'best_score': early_stopping_handler.best_score,
            }
        
        torch.save(checkpoint, filepath)