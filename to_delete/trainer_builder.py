"""Trainer builder for orchestrating training setup."""

from pathlib import Path
from src.builders.base import BaseBuilder
from src.builders.evaluator_builder import EvaluatorBuilder
from src.training.engine import create_train_engine
from src.training.bnn_engine import create_bnn_train_engine
from src.training.handlers.bnn import SampleCollector
from src.training.handlers import (
    attach_progress_bar_to_engine,
    attach_evaluator_handler,
    attach_wandb_logger_to_trainer,
    attach_wandb_logger_to_evaluator,
    attach_checkpoint_handler_to_evaluator,
    attach_last_checkpoint_handler,
    attach_early_stopping,
    attach_scheduler_handler
)


class TrainerBuilder(BaseBuilder):
    """Builds and configures training engine (handles standard and BNN)."""
    
    def build(self, model, loaders, criterion, optimizer, device, output_dir, scheduler=None):
        """
        Build configured training engine with evaluators and handlers.
        
        Automatically detects BNN optimizer and uses appropriate components.
        
        Args:
            model: PyTorch model
            loaders: Dict mapping split names to DataLoaders
            criterion: Loss function
            optimizer: PyTorch optimizer (standard or BNN)
            device: Device to run on
            output_dir: Output directory for checkpoints
            scheduler: Optional learning rate scheduler
        
        Returns:
            trainer: Configured training engine
        """
        # Check if the optimizer is a BNN optimizer to determine which engine to use
        is_bnn = hasattr(optimizer, 'transform')
        
        trainer = self._create_trainer_engine(model, optimizer, criterion, device, is_bnn)
        
        attach_progress_bar_to_engine(trainer, show_loss=True)
        if self.config.wandb.enabled:
            attach_wandb_logger_to_trainer(trainer, self.config.wandb.log_interval)
        
        evaluators = self._build_evaluators(trainer, model, criterion, device, loaders, optimizer)
        
        early_stopping_handler = self._attach_training_handlers(trainer, scheduler, evaluators)
        
        self._attach_checkpointing(trainer, model, optimizer, evaluators, output_dir, 
                                   scheduler, early_stopping_handler, is_bnn)
        
        self._setup_bnn_sampling(trainer, model, output_dir, is_bnn)
        
        return trainer
    
    def _create_trainer_engine(self, model, optimizer, criterion, device, is_bnn):
        """Create training engine (BNN or standard)."""
        if is_bnn:
            sampling_enabled = hasattr(self.config, 'sampling') and self.config.sampling.enabled
            if not sampling_enabled:
                print("Warning: BNN training without sampling - no ensemble evaluation.")
            return create_bnn_train_engine(model, optimizer, device)
        else:
            return create_train_engine(model, optimizer, criterion, device)
    
    def _build_evaluators(self, trainer, model, criterion, device, loaders, optimizer):
        """Build and configure evaluators for all splits."""
        evaluators = {}
        for split_name, eval_config in self.config.evaluation.items():
            if split_name not in loaders:
                print(f"Warning: '{split_name}' not in loaders. Skipping.")
                continue
            
            sample_files = getattr(trainer, 'sample_collector', None)
            sample_files = sample_files.sample_files if sample_files else None
            
            evaluator = EvaluatorBuilder(eval_config.metrics).build(
                model, criterion, device, sample_files=sample_files
            )
            
            evaluators[split_name] = evaluator
            
            attach_progress_bar_to_engine(evaluator, show_loss=False)
            attach_evaluator_handler(trainer, evaluator, loaders[split_name], split_name, eval_config.interval)
            
            if self.config.wandb.enabled:
                attach_wandb_logger_to_evaluator(evaluator, trainer, split_name, optimizer)
        
        return evaluators
    
    def _attach_training_handlers(self, trainer, scheduler, evaluators):
        """Attach handlers to trainer (scheduler, early stopping)."""
        if scheduler is not None:
            attach_scheduler_handler(trainer, scheduler)
        
        early_stopping_handler = None
        if self.config.early_stopping.enabled:
            early_stopping_handler = attach_early_stopping(
                evaluators[self.config.early_stopping.dataset],
                trainer,
                self.config.early_stopping.metric,
                self.config.early_stopping.patience,
                self.config.early_stopping.min_delta,
                self.config.early_stopping.objective
            )
        trainer.early_stopping_handler = early_stopping_handler
        
        return early_stopping_handler
    
    def _attach_checkpointing(self, trainer, model, optimizer, evaluators, output_dir, 
                             scheduler, early_stopping_handler, is_bnn):
        """Attach checkpoint handlers."""
        if not self.config.checkpoint.enabled:
            return
        
        checkpoint_split = self.config.checkpoint.dataset
        
        if checkpoint_split not in evaluators:
            print(f"Warning: Checkpoint on '{checkpoint_split}' but evaluation disabled.")
            return
        
        best_checkpoint_path = Path(output_dir) / "best_model.pt"
        attach_checkpoint_handler_to_evaluator(
            evaluators[checkpoint_split], model, trainer, optimizer,
            self.config.checkpoint.metric, self.config.checkpoint.objective,
            best_checkpoint_path, scheduler=scheduler,
            early_stopping_handler=early_stopping_handler
        )
        
        last_checkpoint_path = Path(output_dir) / "last_checkpoint.pt"
        sample_collector = getattr(trainer, 'sample_collector', None) if is_bnn else None
        attach_last_checkpoint_handler(
            trainer, model, optimizer, last_checkpoint_path, scheduler,
            sample_collector=sample_collector, early_stopping_handler=early_stopping_handler
        )
    
    def _setup_bnn_sampling(self, trainer, model, output_dir, is_bnn):
        """Setup BNN sample collector if enabled."""
        if is_bnn and hasattr(self.config, 'sampling') and self.config.sampling.enabled:
            sample_collector = SampleCollector(output_dir, self.config.sampling)
            sample_collector.attach(trainer, model)
            trainer.sample_collector = sample_collector
        else:
            trainer.sample_collector = None
