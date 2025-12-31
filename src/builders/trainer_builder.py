"""Trainer builder for orchestrating training setup."""

from pathlib import Path
from src.builders.base import BaseBuilder
from src.builders.evaluator_builder import EvaluatorBuilder
from src.training.engine import create_train_engine
from src.training.handlers import (
    attach_progress_bar_to_engine,
    attach_evaluator_handler,
    attach_wandb_logger_to_trainer,
    attach_wandb_logger_to_evaluator,
    attach_checkpoint_handler_to_evaluator
)


class TrainerBuilder(BaseBuilder):
    """Builds and configures Ignite training engine."""
    
    def build(self, model, loaders, criterion, optimizer, device, output_dir):
        """
        Build configured training engine with evaluators and handlers.
        
        Args:
            model: PyTorch model
            loaders: Dict mapping split names to DataLoaders
            criterion: Loss function
            optimizer: PyTorch optimizer
            device: Device to run on
            output_dir: Output directory for checkpoints
        
        Returns:
            trainer: Configured training engine
        """
        # Create training engine
        trainer = create_train_engine(model, optimizer, criterion, device)
        
        # Attach progress bar to trainer
        attach_progress_bar_to_engine(trainer, show_loss=True)
        
        # Attach wandb logging for training if enabled
        if self.config.wandb.enabled:
            attach_wandb_logger_to_trainer(trainer, self.config.wandb.log_interval)
        
        # Loop through configured evaluation splits
        evaluators = {}
        for split_name, eval_config in self.config.evaluation.items():
            if eval_config.interval == 0:
                continue  # Skip disabled evaluations
            
            if split_name not in loaders:
                print(f"Warning: Split '{split_name}' configured for evaluation but not found in loaders. Skipping.")
                continue
            
            # Build evaluator for this split
            evaluator = EvaluatorBuilder(eval_config.metrics).build(
                model, criterion, device
            )
            evaluators[split_name] = evaluator
            
            # Attach progress bar to evaluator
            attach_progress_bar_to_engine(evaluator, show_loss=False)
            
            # Attach evaluation handler
            attach_evaluator_handler(
                trainer, evaluator, loaders[split_name],
                split_name, eval_config.interval
            )
            
            # Attach W&B logging if enabled
            if self.config.wandb.enabled:
                attach_wandb_logger_to_evaluator(evaluator, trainer, split_name)
        
        # Attach checkpointing if enabled
        if self.config.checkpoint.enabled:
            checkpoint_split = self.config.checkpoint.dataset
            
            if checkpoint_split not in evaluators:
                print(f"Warning: Checkpoint configured for '{checkpoint_split}' but evaluation is disabled for it.")
            else:
                checkpoint_path = Path(output_dir) / self.config.checkpoint.filename
                attach_checkpoint_handler_to_evaluator(
                    evaluators[checkpoint_split],
                    model,
                    self.config.checkpoint.metric,
                    checkpoint_path
                )
        
        return trainer
