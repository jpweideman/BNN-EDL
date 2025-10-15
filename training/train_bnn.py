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
import torch
import torch.nn as nn
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.standard_bnn import StandardBNN
from models.base_bnn import create_mlp, create_cnn, create_resnet20
from datasets.classification import get_mnist_dataloaders, get_cifar10_dataloaders, infer_dataset_info
from utils.training import set_random_seeds


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
    """Save BNN model state (final save with any remaining samples)."""
    # Trigger final save of any remaining samples
    if hasattr(bnn, 'posterior_samples') and len(bnn.posterior_samples) > 0:
        bnn._batch_save_samples_to_model()
        bnn.posterior_samples = []  # Clear buffer
    
    # The model.pt file should already exist from incremental saves during training
    model_path = save_path / "model.pt"
    if model_path.exists():
        print(f"Model with incremental samples saved at: {model_path}")
        # Load to check sample count
        try:
            model_state = torch.load(model_path, map_location='cpu')
            sample_count = len(model_state.get('posterior_samples', []))
            print(f"Total posterior samples in model.pt: {sample_count}")
        except:
            print("Could not verify sample count in saved model")
    else:
        print(f"Warning: Expected model.pt not found at {model_path}")


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
        bnn.sample_count = checkpoint.get('sample_count', len(bnn.posterior_samples))
        bnn.model_save_path = str(model_path)  # Set path for future saves
        print(f"Model loaded with {len(bnn.posterior_samples)} posterior samples")
    else:
        print("Model loaded without posterior samples")
    
    return bnn


def get_model_architecture(arch_name: str, input_shape: tuple, num_classes: int, config: dict = None) -> nn.Module:
    """Get model architecture by name with optional configuration."""
    if config is None:
        config = {}
        
    if arch_name == "mlp":
        input_size = np.prod(input_shape)
        hidden_sizes = config.get('mlp_hidden_sizes', [128, 64])
        dropout_rate = config.get('dropout_rate', 0.1)
        return create_mlp(input_size, hidden_sizes, num_classes, dropout_rate)
        
    elif arch_name == "cnn":
        # CNN configuration parameters
        conv_channels = config.get('cnn_conv_channels', None)  # e.g., [32, 64, 128]
        conv_layers_per_block = config.get('cnn_conv_layers_per_block', 2)
        fc_hidden_sizes = config.get('cnn_fc_hidden_sizes', None)  # e.g., [512, 256]
        kernel_size = config.get('cnn_kernel_size', 3)
        pool_size = config.get('cnn_pool_size', 2)
        dropout_rate = config.get('dropout_rate', 0.25)
        use_batch_norm = config.get('cnn_use_batch_norm', False)
        
        return create_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            conv_channels=conv_channels,
            conv_layers_per_block=conv_layers_per_block,
            fc_hidden_sizes=fc_hidden_sizes,
            kernel_size=kernel_size,
            pool_size=pool_size,
            use_batch_norm=use_batch_norm
        )
        
    elif arch_name == "resnet20":
        return create_resnet20(input_shape, num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int = 0, architecture: str = "mlp", 
                   augment: bool = True):
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
            flatten=flatten_images,
        )
    elif dataset_name == "cifar10":
        return get_cifar10_dataloaders(
            data_dir=str(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=1.0,  # Use all training data
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_bnn(config: dict, exp_dir: Path):
    """Main training function."""
    # Set random seeds for reproducibility
    set_random_seeds(config['seed'])
    
    print(f"Starting BNN training with config:")
    print(json.dumps(config, indent=2))
    
    # Get dataloaders
    train_loader, _, test_loader = get_dataloaders(
        config['dataset'], 
        config['batch_size'], 
        config['num_workers'],
        config['architecture'],
        augment=config.get('augment', True)  # Use augment from config, default True
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
    
    # Set model save path for incremental saving during training
    bnn.model_save_path = str(exp_dir / "model.pt")
    
    # Train the model 
    bnn.fit(
        train_loader=train_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        num_burn_in=config['num_burn_in'],
        verbose=True,
        # Pass MCMC-specific parameters
        beta=config['mcmc_beta'],
        alpha=config['mcmc_alpha'],
        sigma=config['mcmc_sigma'],
        xi=config['mcmc_xi'],
        momenta=config['mcmc_momenta']
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
        'training_config': config
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
                       choices=["mlp", "cnn", "resnet20"], 
                       help="Model architecture")
    
    # Data preprocessing
    parser.add_argument("--augment", action="store_true", default=True,
                       help="Enable data augmentation (default: True)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                       help="Disable data augmentation")
    
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
    parser.add_argument("--cnn_use_batch_norm", action="store_true",
                       help="Use batch normalization in CNN")
    parser.add_argument("--dropout_rate", type=float, default=None,
                       help="Dropout rate (default varies by architecture)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_burn_in", type=int, default=50, help="Number of burn-in epochs")
    
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
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    
    # Experiment management
    parser.add_argument("--experiment_name", type=str, default="bnn_experiment", 
                       help="Name for the experiment")
    # Use project root experiments directory
    project_root = Path(__file__).parent.parent
    default_exp_dir = project_root / "experiments"
    parser.add_argument("--output_dir", type=str, default=str(default_exp_dir), 
                       help="Directory to save experiments")
    
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
    
    # Create config dictionary
    config = {
        'dataset': args.dataset,
        'architecture': args.architecture,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_burn_in': args.num_burn_in,
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
        'seed': args.seed,
        'experiment_name': args.experiment_name,
        'timestamp': datetime.datetime.now().isoformat(),
        
        # Architecture-specific parameters
        'mlp_hidden_sizes': args.mlp_hidden_sizes,
        'cnn_conv_channels': args.cnn_conv_channels,
        'cnn_conv_layers_per_block': args.cnn_conv_layers_per_block,
        'cnn_fc_hidden_sizes': args.cnn_fc_hidden_sizes,
        'cnn_kernel_size': args.cnn_kernel_size,
        'cnn_pool_size': args.cnn_pool_size,
        'cnn_use_batch_norm': args.cnn_use_batch_norm,
        'dropout_rate': args.dropout_rate
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
        print(f"Posterior Samples: {int(results['test_metrics']['num_samples'])}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
