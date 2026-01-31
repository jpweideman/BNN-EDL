"""Main training script."""

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.setup.data_loader import DataLoaderSetup
from src.setup.checkpoint import CheckpointSetup
from src.setup.evaluator import EvaluatorSetup
from src.setup.trainer import TrainerSetup
from src.setup.weight_loader import WeightLoaderSetup
from src.builders.model_builder import ModelBuilder
from src.builders.loss_builder import LossBuilder
from src.builders.likelihood_builder import LikelihoodBuilder
from src.builders.prior_builder import PriorBuilder
from src.builders.optimizer_builder import OptimizerBuilder
from src.builders.scheduler_builder import SchedulerBuilder
from src.utils import set_seed, setup_device


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    set_seed(cfg.seed)
    device = setup_device(cfg.training.device)
    
    # Build data loaders and model
    loaders = DataLoaderSetup(cfg.dataset).create_loaders(seed=cfg.seed)
    model = ModelBuilder(cfg.model).build().to(device)
    
    # Initialize checkpoint setup
    checkpoint_setup = CheckpointSetup(output_dir, cfg.training.checkpoint)
    
    # Build optimizer and other components
    criterion = LossBuilder(cfg.training.loss).build()
    likelihood_fn = LikelihoodBuilder(cfg.training.likelihood).build() if hasattr(cfg.training, 'likelihood') else None
    dataset_size = len(loaders['train'].dataset)
    prior_fn = PriorBuilder(cfg.training.prior).build(num_data=dataset_size) if hasattr(cfg.training, 'prior') else None
    optimizer = OptimizerBuilder(cfg.training.optimizer).build(
        model.parameters(),
        model=model,
        loss_fn=criterion,
        likelihood_fn=likelihood_fn,
        prior_fn=prior_fn,
        num_data=dataset_size
    )
    
    # Build scheduler if enabled
    scheduler = SchedulerBuilder(cfg.training.scheduler).build(optimizer) if cfg.training.scheduler.enabled else None
    
    # Load checkpoint (includes scheduler step when resuming)
    start_epoch, sample_files, early_stopping_state = checkpoint_setup.load_checkpoint(
        model, optimizer, device, scheduler
    )
    
    # Initialize W&B
    checkpoint_setup.init_wandb(cfg.training.wandb, cfg)
    
    # Prepare configs 
    pretrained_config = cfg.training.pretrained if hasattr(cfg.training, 'pretrained') and cfg.training.pretrained.enabled else None
    sampling_config = cfg.training.sampling if hasattr(cfg.training, 'sampling') and cfg.training.sampling.enabled else None
    checkpoint_config = cfg.training.checkpoint if cfg.training.checkpoint.enabled else None
    wandb_config = cfg.training.wandb if cfg.training.wandb.enabled else None
    early_stopping_config = cfg.training.early_stopping if cfg.training.early_stopping.enabled else None
    
    # Load pretrained weights if enabled
    if pretrained_config is not None:
        WeightLoaderSetup().load_pretrained_weights(model, pretrained_config.path, device)
    
    # Build evaluators 
    evaluators = EvaluatorSetup(cfg.training.evaluation).create_evaluators(
        model, criterion, device, loaders,
        optimizer=optimizer
    )
    
    # Build trainer
    trainer = TrainerSetup().create_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        evaluators=evaluators,
        scheduler=scheduler,
        sampling_config=sampling_config,
        checkpoint_config=checkpoint_config,
        wandb_config=wandb_config,
        early_stopping_config=early_stopping_config
    )
    
    # Restore state from checkpoint
    checkpoint_setup.restore_trainer_state(trainer, start_epoch, sample_files, early_stopping_state)
    
    # Train
    if start_epoch < cfg.training.num_epochs:
        trainer.run(loaders['train'], max_epochs=cfg.training.num_epochs)
    else:
        print(f"Training already completed ({start_epoch}/{cfg.training.num_epochs} epochs)")


if __name__ == "__main__":
    main()