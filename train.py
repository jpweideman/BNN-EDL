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
from src.builders.trainer_builder import TrainerBuilder
from src.utils import set_seed, setup_device




@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    set_seed(cfg.seed)
    device = setup_device(cfg.training.device)
    
    if cfg.training.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.training.wandb.project,
                name=cfg.training.wandb.name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=output_dir
            )
        except ImportError:
            print("Warning: wandb not installed")
            cfg.training.wandb.enabled = False
    
    loaders = DataLoaderBuilder(cfg.dataset).build(seed=cfg.seed)
    model = ModelBuilder(cfg.model).build().to(device)
    criterion = LossBuilder(cfg.training.loss).build()
    optimizer = OptimizerBuilder(cfg.training.optimizer).build(model.parameters())
    
    trainer = TrainerBuilder(cfg.training).build(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir
    )
    
    trainer.run(loaders['train'], max_epochs=cfg.training.num_epochs)
    
    torch.save(model.state_dict(), Path(output_dir) / "final_model.pt")
    
    if cfg.training.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()

