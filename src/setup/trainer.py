"""Trainer setup for orchestrating training pipeline."""

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
    attach_scheduler_handler
)


class TrainerSetup:
    """Orchestrates training pipeline setup."""
    
    def create_trainer(self, model, optimizer, criterion, device, output_dir, evaluators,
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
            evaluators: Dict of evaluators from EvaluatorSetup
            scheduler: Optional learning rate scheduler
            sampling_config: Optional BNN sampling configuration
            checkpoint_config: Optional checkpoint configuration
            wandb_config: Optional W&B configuration
            early_stopping_config: Optional early stopping configuration
        
        Returns:
            Configured training engine
        """
        is_bnn = isinstance(optimizer, BNNOptimizer)
        
        trainer = self._create_trainer_engine(model, optimizer, criterion, device, is_bnn)
        
        attach_progress_bar_to_engine(trainer, show_loss=True)
        if wandb_config is not None:
            attach_wandb_logger_to_trainer(trainer, wandb_config.log_interval)
        
        self._setup_bnn_sampling(trainer, model, output_dir, is_bnn, sampling_config)
        
        self._attach_evaluators(trainer, evaluators, optimizer, wandb_config)
        
        early_stopping_handler = self._attach_training_handlers(trainer, scheduler, evaluators,
                                                                early_stopping_config)
        
        self._attach_checkpointing(trainer, model, optimizer, evaluators, output_dir,
                                  scheduler, early_stopping_handler, checkpoint_config, is_bnn)
        
        return trainer
    
    def _create_trainer_engine(self, model, optimizer, criterion, device, is_bnn):
        """Create training engine (BNN or standard)."""
        if is_bnn:
            return create_bnn_train_engine(model, optimizer, device)
        else:
            return create_train_engine(model, optimizer, criterion, device)
    
    def _attach_evaluators(self, trainer, evaluators, optimizer, wandb_config):
        """Attach evaluators to trainer."""
        for split_name, eval_data in evaluators.items():
            evaluator = eval_data['evaluator']
            
            # Connect BNN evaluator to sampling_manager
            if hasattr(trainer, 'sampling_manager') and trainer.sampling_manager:
                evaluator.sampling_manager = trainer.sampling_manager
            
            attach_evaluator_handler(
                trainer, evaluator, eval_data['loader'],
                split_name, eval_data['interval']
            )
            if wandb_config is not None:
                attach_wandb_logger_to_evaluator(
                    evaluator, trainer, split_name, optimizer
                )
    
    def _attach_training_handlers(self, trainer, scheduler, evaluators, early_stopping_config):
        """Attach scheduler and early stopping handlers to trainer."""
        if scheduler is not None:
            attach_scheduler_handler(trainer, scheduler)
        
        early_stopping_handler = None
        if early_stopping_config is not None:
            early_stopping_handler = attach_early_stopping(
                evaluators[early_stopping_config.dataset]['evaluator'],
                trainer,
                early_stopping_config.metric,
                early_stopping_config.patience,
                early_stopping_config.min_delta,
                early_stopping_config.objective
            )
        trainer.early_stopping_handler = early_stopping_handler
        
        return early_stopping_handler
    
    def _attach_checkpointing(self, trainer, model, optimizer, evaluators, output_dir,
                             scheduler, early_stopping_handler, checkpoint_config, is_bnn):
        """Attach checkpoint handlers."""
        if checkpoint_config is None:
            return
        
        checkpoint_split = checkpoint_config.dataset
        
        if checkpoint_split not in evaluators:
            print(f"Warning: Checkpoint configured for '{checkpoint_split}' but no evaluator found. Checkpointing disabled.")
            return
        
        best_checkpoint_path = Path(output_dir) / "best_model.pt"
        attach_checkpoint_handler_to_evaluator(
            evaluators[checkpoint_split]['evaluator'], model, trainer, optimizer,
            checkpoint_config.metric, checkpoint_config.objective,
            best_checkpoint_path, scheduler=scheduler,
            early_stopping_handler=early_stopping_handler
        )
        
        last_checkpoint_path = Path(output_dir) / "last_checkpoint.pt"
        sampling_manager = getattr(trainer, 'sampling_manager', None) if is_bnn else None
        attach_last_checkpoint_handler(
            trainer, model, optimizer, last_checkpoint_path, scheduler,
            sampling_manager=sampling_manager, early_stopping_handler=early_stopping_handler
        )
    
    def _setup_bnn_sampling(self, trainer, model, output_dir, is_bnn, sampling_config):
        """Setup BNN sampling manager if enabled."""
        sampling_manager = None
        if is_bnn and sampling_config is not None:
            from src.training.handlers.bnn import SamplingManager
            sampling_manager = SamplingManager(output_dir, sampling_config)
            sampling_manager.attach(trainer, model)
        elif is_bnn:
            print("Warning: BNN training without sampling. No ensemble evaluation possible.")
        trainer.sampling_manager = sampling_manager
