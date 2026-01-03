"""Main training script."""

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch
from pathlib import Path

from src.builders.data_builder import DataLoaderBuilder
from src.builders.model_builder import ModelBuilder
from src.builders.loss_builder import LossBuilder
from src.builders.optimizer_builder import OptimizerBuilder
from src.builders.scheduler_builder import SchedulerBuilder
from src.builders.trainer_builder import TrainerBuilder
from src.utils import set_seed, setup_device, CheckpointManager




@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    set_seed(cfg.seed)
    device = setup_device(cfg.training.device)
    
    loaders = DataLoaderBuilder(cfg.dataset).build(seed=cfg.seed)
    model = ModelBuilder(cfg.model).build().to(device)
    criterion = LossBuilder(cfg.training.loss).build()
    optimizer = OptimizerBuilder(cfg.training.optimizer).build(model.parameters())
    
    scheduler = SchedulerBuilder(cfg.training.scheduler).build(optimizer) if cfg.training.scheduler is not None else None
    
    # Initialize checkpoint manager and load checkpoint if exists
    checkpoint_manager = CheckpointManager(output_dir, cfg.training.checkpoint)
    start_epoch = checkpoint_manager.load_checkpoint(model, optimizer, device, scheduler)
    
    # Step scheduler to update LR for the current epoch when resuming
    if scheduler is not None and start_epoch > 0:
        scheduler.step()
    
    # Initialize W&B. Checkpoint manager handles resumption
    wandb_enabled = checkpoint_manager.init_wandb(cfg.training.wandb, cfg)
    
    trainer = TrainerBuilder(cfg.training).build(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir
    )
    
    # Train for remaining epochs
    remaining_epochs = cfg.training.num_epochs - start_epoch
    if remaining_epochs > 0:
        trainer.run(loaders['train'], max_epochs=remaining_epochs)
    else:
        print(f"Training already completed ({start_epoch}/{cfg.training.num_epochs} epochs)")
    
    if wandb_enabled:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()

