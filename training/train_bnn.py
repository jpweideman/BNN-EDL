#!/usr/bin/env python3
"""
Training script for Bayesian Neural Networks.

This script provides a comprehensive training pipeline for BNN models with:
- Model saving/loading
- Configuration management
- Multiple dataset support
- Evaluation and metrics logging
- Experiment tracking
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
import warnings
import torch
import torch.nn as nn

# Suppress Pydantic warnings from posteriors library
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.standard_bnn import StandardBNN
from models.base_bnn import create_mlp, create_cnn, create_resnet20, create_resnet18
from datasets.classification import get_mnist_dataloaders, get_cifar10_dataloaders, infer_dataset_info


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(config: dict, save_path: Path):
    """Save experiment configuration."""
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_model(bnn: StandardBNN, save_path: Path):
    """Save BNN model state."""
    model_state = {
        'model_state_dict': bnn.model.state_dict(),
        'posterior_samples': bnn.posterior_samples if hasattr(bnn, 'posterior_samples') else [],
        'num_classes': bnn.num_classes,
        'prior_std': bnn.prior_std,
        'temperature': bnn.temperature,
        'mcmc_method': bnn.mcmc_method,
        'device': str(bnn.device)
    }
    torch.save(model_state, save_path / "model.pt")
    print(f"Model saved to {save_path / 'model.pt'}")


def load_model(model_path: Path, model_architecture: nn.Module) -> StandardBNN:
    """Load BNN model from saved state."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Recreate BNN
    bnn = StandardBNN(
        model=model_architecture,
        num_classes=checkpoint['num_classes'],
        prior_std=checkpoint['prior_std'],
        temperature=checkpoint['temperature'],
        mcmc_method=checkpoint['mcmc_method'],
        device=checkpoint['device']
    )
    
    # Load model weights
    bnn.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load posterior samples if available
    if 'posterior_samples' in checkpoint and checkpoint['posterior_samples']:
        bnn.posterior_samples = checkpoint['posterior_samples']
    
    return bnn


def get_model_architecture(arch_name: str, input_shape: tuple, num_classes: int, config: dict = None) -> nn.Module:
    """Get model architecture by name with optional configuration."""
    if config is None:
        config = {}
        
    if arch_name == "mlp":
        input_size = np.prod(input_shape)
        hidden_sizes = config.get('mlp_hidden_sizes', [128, 64])
        return create_mlp(input_size, hidden_sizes, num_classes)
        
    elif arch_name == "cnn":
        # CNN configuration parameters
        conv_channels = config.get('cnn_conv_channels', None)  # e.g., [32, 64, 128]
        conv_layers_per_block = config.get('cnn_conv_layers_per_block', 2)
        fc_hidden_sizes = config.get('cnn_fc_hidden_sizes', None)  # e.g., [512, 256]
        kernel_size = config.get('cnn_kernel_size', 3)
        pool_size = config.get('cnn_pool_size', 2)
        
        return create_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_channels=conv_channels,
            conv_layers_per_block=conv_layers_per_block,
            fc_hidden_sizes=fc_hidden_sizes,
            kernel_size=kernel_size,
            pool_size=pool_size
        )
        
    elif arch_name == "resnet20":
        return create_resnet20(input_shape, num_classes)
        
    elif arch_name == "resnet18":
        return create_resnet18(input_shape, num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int = 0, architecture: str = "mlp"):
    """Get dataloaders for specified dataset."""
    # Use project root data directory
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data"
    
    # Determine whether to flatten based on architecture
    # MLPs need flattened input, CNNs and ResNets need 2D/3D input
    flatten_images = (architecture == "mlp")
    
    if dataset_name == "mnist":
        return get_mnist_dataloaders(
            data_dir=str(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=1.0,  # Use all training data
            flatten=flatten_images
        )
    elif dataset_name == "cifar10":
        return get_cifar10_dataloaders(
            data_dir=str(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=1.0  # Use all training data
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_bnn(config: dict, exp_dir: Path):
    """Main training function."""
    print(f"Starting BNN training with config:")
    print(json.dumps(config, indent=2))
    
    # Get dataloaders
    train_loader, _, test_loader = get_dataloaders(
        config['dataset'], 
        config['batch_size'], 
        config['num_workers'],
        config['architecture']
    )
    
    # Infer dataset info
    dataset_info = infer_dataset_info(train_loader)
    print(f"Dataset info: {dataset_info}")
    
    # Create model architecture
    model = get_model_architecture(
        config['architecture'], 
        dataset_info['input_shape'], 
        dataset_info['num_classes'],
        config
    )
    
    # Create BNN
    bnn = StandardBNN(
        model=model,
        num_classes=dataset_info['num_classes'],
        prior_std=config['prior_std'],
        temperature=config['temperature'],
        mcmc_method=config['mcmc_method'],
        device=config['device']
    )
    
    print(f"Created {config['mcmc_method']} BNN with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {bnn.device}")
    
    # Train the model 
    bnn.fit(
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        num_burn_in=config['num_burn_in'],
        verbose=True,
        eval_frequency=config.get('eval_frequency', 20),
        test_loader=test_loader if not config.get('no_periodic_eval', False) else None,
        # Weights & Biases parameters
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project', 'bnn-training'),
        wandb_run_name=config['timestamped_exp_name'],
        wandb_config=config,  # Pass full config to wandb 
        # Pass MCMC-specific parameters
        beta=config['mcmc_beta'],
        alpha=config['mcmc_alpha'],
        sigma=config['mcmc_sigma'],
        xi=config['mcmc_xi'],
        momenta=config.get('mcmc_momenta', None)
    )
    
    # Evaluate on test set
    test_metrics = bnn.evaluate(test_loader)
    print("Test Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'dataset_info': dataset_info,
        'test_metrics': test_metrics,
        'training_config': config,
        'training_history': {
            'log_posterior_history': bnn.log_posterior_history,
            'training_loss_history': bnn.training_loss_history,
            'test_metrics_history': bnn.test_metrics_history
        }
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    save_model(bnn, exp_dir)
    
    print(f"\nExperiment completed. Results saved to {exp_dir}")
    return bnn, results


def main():
    parser = argparse.ArgumentParser(description="Train Bayesian Neural Networks")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, default="mnist", 
                       choices=["mnist", "cifar10"], help="Dataset to use")
    parser.add_argument("--architecture", type=str, default="mlp",
                       choices=["mlp", "cnn", "resnet20", "resnet18"], 
                       help="Model architecture")
    
    # Architecture-specific parameters
    parser.add_argument("--mlp_hidden_sizes", type=int, nargs='+', default=[128, 64],
                       help="Hidden layer sizes for MLP (e.g., --mlp_hidden_sizes 256 128 64)")
    parser.add_argument("--cnn_conv_channels", type=int, nargs='+', default=None,
                       help="Conv channel sizes for CNN (e.g., --cnn_conv_channels 32 64 128)")
    parser.add_argument("--cnn_conv_layers_per_block", type=int, default=2,
                       help="Number of conv layers per block in CNN")
    parser.add_argument("--cnn_fc_hidden_sizes", type=int, nargs='+', default=None,
                       help="FC layer sizes for CNN (e.g., --cnn_fc_hidden_sizes 512 256)")
    parser.add_argument("--cnn_kernel_size", type=int, default=3,
                       help="Kernel size for CNN convolutions")
    parser.add_argument("--cnn_pool_size", type=int, default=2,
                       help="Pool size for CNN max pooling")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    # MCMC parameters
    parser.add_argument("--num_burn_in", type=int, default=50, help="Number of burn-in epochs")
    parser.add_argument("--eval_frequency", type=int, default=20, help="Frequency of test evaluation (every N epochs after burn-in)")
    parser.add_argument("--no_periodic_eval", action="store_true", help="Disable periodic test evaluation during training")
    
    # BNN parameters
    parser.add_argument("--prior_std", type=float, default=1.0, help="Prior standard deviation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for posterior")
    parser.add_argument("--mcmc_method", type=str, default="sgld", 
                       choices=["sgld", "sghmc", "sgnht", "baoa"], help="MCMC method")
    
    # MCMC-specific parameters
    parser.add_argument("--mcmc_beta", type=float, default=0.0, 
                       help="Gradient noise coefficient for all methods")
    parser.add_argument("--mcmc_alpha", type=float, default=0.01, 
                       help="Friction coefficient for SGHMC/SGNHT/BAOA")
    parser.add_argument("--mcmc_sigma", type=float, default=1.0, 
                       help="Standard deviation of momenta target distribution for SGHMC/SGNHT/BAOA")
    parser.add_argument("--mcmc_xi", type=float, default=None, 
                       help="Initial thermostat value for SGNHT (defaults to alpha if not set)")
    parser.add_argument("--mcmc_momenta", type=str, default=None, 
                       help="Initial momenta for SGHMC/SGNHT/BAOA (None for random initialization)")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Experiment management
    parser.add_argument("--experiment_name", type=str, default="bnn_experiment", 
                       help="Name for the experiment")
    # Use project root experiments directory
    project_root = Path(__file__).parent.parent
    default_exp_dir = project_root / "experiments"
    parser.add_argument("--output_dir", type=str, default=str(default_exp_dir), 
                       help="Directory to save experiments")
    
    # Weights & Biases integration
    parser.add_argument("--use_wandb", action="store_true", 
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="bnn-training",
                       help="W&B project name")
    
    # Quick presets
    parser.add_argument("--quick", action="store_true", 
                       help="Quick training (fewer epochs for testing)")
    
    args = parser.parse_args()
    
    # Apply quick preset
    if args.quick:
        args.num_epochs = 10
        args.num_burn_in = 5
        print("Quick mode: Using reduced epochs for testing")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    
    # Get the timestamped directory name to use as W&B run name
    timestamped_exp_name = exp_dir.name
    
    # Create config dictionary
    config = {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_burn_in': args.num_burn_in,
        'eval_frequency': args.eval_frequency,
        'no_periodic_eval': args.no_periodic_eval,
        'prior_std': args.prior_std,
        'temperature': args.temperature,
        'mcmc_method': args.mcmc_method,
        'mcmc_beta': args.mcmc_beta,
        'mcmc_alpha': args.mcmc_alpha,
        'mcmc_sigma': args.mcmc_sigma,
        'mcmc_xi': args.mcmc_xi if args.mcmc_xi is not None else args.mcmc_alpha,
        'mcmc_momenta': args.mcmc_momenta,
        'device': args.device,
        'num_workers': args.num_workers,
        'experiment_name': args.experiment_name,
        'experiment_dir': str(exp_dir),  
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'timestamped_exp_name': timestamped_exp_name,
        
        # Architecture-specific parameters
        'mlp_hidden_sizes': args.mlp_hidden_sizes,
        'cnn_conv_channels': args.cnn_conv_channels,
        'cnn_conv_layers_per_block': args.cnn_conv_layers_per_block,
        'cnn_fc_hidden_sizes': args.cnn_fc_hidden_sizes,
        'cnn_kernel_size': args.cnn_kernel_size,
        'cnn_pool_size': args.cnn_pool_size,
    }
    
    # Save config
    save_config(config, exp_dir)
    
    # Train model
    try:
        bnn, results = train_bnn(config, exp_dir)
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {exp_dir}")
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test ECE: {results['test_metrics']['ece']:.4f}")
        print(f"Posterior Samples: {int(results['test_metrics']['num_posterior_samples'])}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
