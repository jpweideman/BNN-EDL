import torch
from pathlib import Path


class CheckpointManager:
    """Manages checkpoint saving/loading and W&B resumption for training."""
    
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
            int: Epoch number to resume from (0 if no checkpoint loaded)
        """
        checkpoint_path = self.output_dir / "last_checkpoint.pt"
        
        if not checkpoint_path.exists():
            print(f"No checkpoint found, starting new training")
            self.start_epoch = 0
            self.wandb_run_id = None
            return 0
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.wandb_run_id = checkpoint.get('wandb_run_id', None)
    
        return self.start_epoch
    
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
        
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed")
            return False
        
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
                    metric_name = f"{split}_{metric_config.name}"
                    summary_type = "max" if metric_config.mode == "maximize" else "min"
                    wandb.define_metric(metric_name, summary=summary_type)
        
        return True

