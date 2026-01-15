"""Checkpoint orchestrator for loading checkpoints and managing W&B resumption."""

import torch
from pathlib import Path
from src.utils.objective import to_wandb_summary


class CheckpointSetup:
    """Orchestrates checkpoint loading and W&B resumption for training."""
    
    def __init__(self, output_dir, checkpoint_config):
        """
        Args:
            output_dir: Directory to save/load checkpoints
            checkpoint_config: Checkpoint configuration from Hydra
        """
        self.output_dir = Path(output_dir)
        self.config = checkpoint_config
        self.wandb_run_id = None
        self.start_epoch = 0
    
    def load_checkpoint(self, model, optimizer, device, scheduler=None):
        """
        Load the last checkpoint if it exists in output_dir.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            device: Device to map checkpoint to
            scheduler: Optional scheduler to load state into
        
        Returns:
            tuple: (start_epoch, sample_files, early_stopping_state)
                - start_epoch: Epoch to resume from
                - sample_files: List of sample files to reload (None if not BNN)
                - early_stopping_state: Early stopping state dict (None if not saved)
        """
        checkpoint_path = self.output_dir / "last_checkpoint.pt"
        
        if not checkpoint_path.exists():
            print(f"No checkpoint found, starting new training")
            self.start_epoch = 0
            self.wandb_run_id = None
            return 0, None, None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore RNG state for reproducibility
        torch.set_rng_state(checkpoint['rng_state']['torch'].cpu())
        if checkpoint['rng_state']['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['torch_cuda'])
    
        self.start_epoch = checkpoint['epoch']
        self.wandb_run_id = checkpoint.get('wandb_run_id', None)
        sample_files = checkpoint.get('sample_files', None)
        early_stopping_state = checkpoint.get('early_stopping_state', None)
        
        # Handle scheduler restoration
        if scheduler is not None and self.start_epoch > 0:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print(f"Scheduler configured but not in checkpoint. Stepping {self.start_epoch} times to catch up.")
                for _ in range(self.start_epoch):
                    scheduler.step()
    
        return self.start_epoch, sample_files, early_stopping_state
    
    def restore_trainer_state(self, trainer, start_epoch, sample_files, early_stopping_state):
        """
        Restore trainer state after checkpoint loading.
        
        Args:
            trainer: Training engine to restore state to
            start_epoch: Epoch to resume from
            sample_files: List of sample files for BNN
            early_stopping_state: Early stopping state dict
        """
        if start_epoch > 0:
            trainer.state.epoch = start_epoch
            trainer.state.max_epochs = start_epoch  
        
        if sample_files and trainer.sample_collector is not None:
            trainer.sample_collector.sample_files = sample_files
        
        if early_stopping_state is not None and trainer.early_stopping_handler is not None:
            trainer.early_stopping_handler.counter = early_stopping_state['counter']
            trainer.early_stopping_handler.best_score = early_stopping_state['best_score']
    
    def init_wandb(self, wandb_config, hydra_config):
        """
        Initialize W&B, resuming existing run if checkpoint was loaded.
        
        Args:
            wandb_config: W&B configuration from Hydra
            hydra_config: Full Hydra config (for logging)
        
        Returns:
            bool: True if W&B was initialized successfully
        """
        if not wandb_config.enabled:
            return False
        import wandb
        
        if self.wandb_run_id:
            # Resume existing W&B run
            wandb.init(
                project=wandb_config.project,
                id=self.wandb_run_id,
                resume="must",
                dir=str(self.output_dir)
            )
            print(f"Resumed W&B run: {self.wandb_run_id}")
        else:
            # Start new W&B run
            from omegaconf import OmegaConf
            wandb.init(
                project=wandb_config.project,
                name=wandb_config.name,
                config=OmegaConf.to_container(hydra_config, resolve=True),
                dir=str(self.output_dir)
            )
        
        # Define metric summaries from config
        if hasattr(wandb_config, 'summary_metrics'):
            for split, metrics in wandb_config.summary_metrics.items():
                for metric_config in metrics:
                    summary_type = to_wandb_summary(metric_config.objective)
                    metric_name = f"{split}_{metric_config.name}"
                    wandb.define_metric(metric_name, summary=summary_type)
        
        return True

