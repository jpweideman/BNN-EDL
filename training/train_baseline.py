#!/usr/bin/env python3
"""
Baseline training script for standard neural networks.

Trains standard (non-Bayesian) models using the same architectures as the BNN
to provide a comparison baseline for evaluating BNN performance.

Supports:
- Model architectures: MLP, CNN, ResNet20, ResNet18
- Datasets: MNIST, CIFAR-10
- Standard SGD with learning rate schedules
- Early stopping, checkpointing, and metrics logging
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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.base_bnn import create_mlp, create_cnn, create_resnet20, create_resnet18
from datasets.classification import get_mnist_dataloaders, get_cifar10_dataloaders, infer_dataset_info

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


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


def get_model_architecture(architecture: str, input_shape: tuple, num_classes: int, config: dict) -> nn.Module:
    """Create model architecture."""
    if architecture == "mlp":
        hidden_sizes = config.get('mlp_hidden_sizes', [128, 64])
        return create_mlp(int(np.prod(input_shape)), hidden_sizes, num_classes)
    elif architecture == "cnn":
        conv_channels = config.get('cnn_conv_channels', [32, 64])
        fc_hidden_sizes = config.get('cnn_fc_hidden_sizes', [512])
        conv_layers_per_block = config.get('cnn_conv_layers_per_block', 2)
        kernel_size = config.get('cnn_kernel_size', 3)
        pool_size = config.get('cnn_pool_size', 2)
        return create_cnn(input_shape, conv_channels, fc_hidden_sizes, num_classes,
                         conv_layers_per_block, kernel_size, pool_size)
    elif architecture == "resnet20":
        return create_resnet20(input_shape, num_classes)
    elif architecture == "resnet18":
        return create_resnet18(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int = 0, architecture: str = "mlp"):
    """Get data loaders for the specified dataset."""
    if dataset_name == "mnist":
        # Get dataloaders with train_split=1.0 to use all training data
        train_loader, _, test_loader = get_mnist_dataloaders(batch_size=batch_size, num_workers=num_workers, train_split=1.0)
        return train_loader, test_loader
    elif dataset_name == "cifar10":
        # Get dataloaders with train_split=1.0 to use all training data
        train_loader, _, test_loader = get_cifar10_dataloaders(batch_size=batch_size, num_workers=num_workers, train_split=1.0)
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_ece(predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        predictions: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    confidences = torch.max(predictions, dim=1)[0]
    pred_classes = torch.argmax(predictions, dim=1)
    accuracies = (pred_classes == labels).float()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def train_baseline(config: dict, exp_dir: Path = None, resume_from: Path = None):
    """Train a standard neural network baseline.
    
    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory (created if None)
        resume_from: Path to experiment directory to resume from
    """
    print(f"Starting baseline training with config:")
    print(json.dumps(config, indent=2))
    
    # Determine which directory to use
    if resume_from:
        resume_from = Path(resume_from)
        if resume_from.exists():
            checkpoint_path = resume_from / "checkpoint.pt"
            if checkpoint_path.exists():
                exp_dir = resume_from   
                print(f"Resuming training from: {exp_dir}")
            else:
                print(f"\n No checkpoint found in {resume_from}")
                print(f"  Starting new run instead\n")
                if exp_dir is None:
                    exp_dir = create_experiment_dir("../experiments", config['experiment_name'])
        else:
            print(f"\n Directory {resume_from} does not exist")
            print(f"  Starting new run instead\n")
            if exp_dir is None:
                exp_dir = create_experiment_dir("../experiments", config['experiment_name'])
    
    # Create experiment directory if not resuming
    if exp_dir is None:
        exp_dir = create_experiment_dir("../experiments", config['experiment_name'])
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        config['dataset'],
        config['batch_size'],
        config['num_workers'],
        config['architecture']
    )
    
    # Infer dataset info
    dataset_info = infer_dataset_info(train_loader)
    print(f"Dataset info: {dataset_info}")
    
    # Create model
    model = get_model_architecture(
        config['architecture'],
        dataset_info['input_shape'],
        dataset_info['num_classes'],
        config
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created {config['architecture']} baseline with {num_params} parameters")
    print(f"Using device: {device}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                         momentum=config.get('momentum', 0.9),
                         weight_decay=config.get('weight_decay', 1e-4))
    
    use_lr_schedule = config.get('use_lr_schedule', True)
    if use_lr_schedule:
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    else:
        scheduler = None
    
    # Check for checkpoint in exp_dir
    checkpoint_path = exp_dir / "checkpoint.pt"
    start_epoch = 0
    best_val_acc = 0.0
    train_loss_history = []
    train_acc_history = []
    test_metrics_history = []
    wandb_run_id = None
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        train_loss_history = checkpoint['train_loss_history']
        train_acc_history = checkpoint['train_acc_history']
        test_metrics_history = checkpoint['test_metrics_history']
        
        # Load wandb run ID if available
        wandb_id_path = exp_dir / "wandb_run_id.txt"
        if wandb_id_path.exists():
            with open(wandb_id_path, 'r') as f:
                wandb_run_id = f.read().strip()
            print(f"Loaded W&B run ID: {wandb_run_id}")
    
    # Setup wandb
    if config.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb_dir = os.path.join(str(exp_dir), "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb_init_kwargs = {
            'project': config.get('wandb_project', 'bnn-training'),
            'name': f"baseline_{config['experiment_name']}",
            'config': config,
            'dir': wandb_dir,
        }
        
        # If resuming and we have a run ID, resume that run instead of creating new one
        if wandb_run_id:
            wandb_init_kwargs['resume'] = 'must'
            wandb_init_kwargs['id'] = wandb_run_id
        else:
            wandb_init_kwargs['reinit'] = True

        wandb.init(**wandb_init_kwargs)
        
        # Save run ID for resumption
        wandb_id_path = exp_dir / "wandb_run_id.txt"
        with open(wandb_id_path, 'w') as f:
            f.write(wandb.run.id)
    
    # Training loop
    epoch_pbar = tqdm(range(start_epoch, config['num_epochs']), desc="Training Baseline")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Flatten for MLP
            if config['architecture'] == 'mlp' and len(images.shape) > 2:
                images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        if scheduler:
            scheduler.step()
        
        # Periodic test evaluation (every eval_frequency epochs, for information only)
        eval_frequency = config.get('eval_frequency', 10)
        if (epoch + 1) % eval_frequency == 0:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    if config['architecture'] == 'mlp' and len(images.shape) > 2:
                        images = images.view(images.size(0), -1)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item() * labels.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.append(probs)
                    all_labels.append(labels)
                    
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += labels.size(0)
            
            test_loss /= test_total
            test_acc = test_correct / test_total
            
            # Compute ECE
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            test_ece = compute_ece(all_predictions, all_labels)
            
            test_metrics = {
                'epoch': epoch + 1,
                'accuracy': test_acc,
                'loss': test_loss,
                'ece': test_ece
            }
            test_metrics_history.append(test_metrics)
            
            if test_acc > best_val_acc:
                best_val_acc = test_acc
                torch.save(model.state_dict(), exp_dir / "best_model.pt")
            
            if config.get('verbose', True):
                print(f"\n[Epoch {epoch+1}] Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}, ECE: {test_ece:.4f}")
            
            # Log to wandb
            if config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'test_ece': test_ece,
                }, step=epoch + 1)
        
        # Log training metrics to wandb
        if config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)
        
        epoch_pbar.set_postfix({
            'train_loss': f"{train_loss:.4f}",
            'train_acc': f"{train_acc:.4f}"
        })
        
        # Save checkpoint after each epoch (for resuming training)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss_history': train_loss_history,
            'train_acc_history': train_acc_history,
            'test_metrics_history': test_metrics_history,
        }
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, exp_dir / "checkpoint.pt")
    
    # Final test evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    print("\nFinal evaluation on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            
            if config['architecture'] == 'mlp' and len(images.shape) > 2:
                images = images.view(images.size(0), -1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            all_predictions.append(probs)
            all_labels.append(labels)
            
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss /= test_total
    test_acc = test_correct / test_total
    
    # Compute ECE
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    test_ece = compute_ece(all_predictions, all_labels)
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  ECE: {test_ece:.4f}")
    
    # Save results
    results = {
        'dataset_info': dataset_info,
        'test_metrics': {
            'accuracy': test_acc,
            'loss': test_loss,
            'ece': test_ece
        },
        'training_config': config,
        'training_history': {
            'train_loss': train_loss_history,
            'train_accuracy': train_acc_history,
            'test_metrics': test_metrics_history,
        }
    }
    
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), exp_dir / "final_model.pt")
    
    print(f"\nExperiment completed. Results saved to {exp_dir}")
    
    # Finish wandb
    if config.get('use_wandb', False) and WANDB_AVAILABLE:
        wandb.finish()
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Neural Networks")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar10"], help="Dataset to use")
    parser.add_argument("--architecture", type=str, default="mlp",
                       choices=["mlp", "cnn", "resnet20", "resnet18"],
                       help="Model architecture")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--eval_frequency", type=int, default=10, help="Frequency of test evaluation (every N epochs)")
    
    # Schedule
    parser.add_argument("--use_lr_schedule", type=lambda x: x.lower() == 'true', default=True,
                       help="Use cosine annealing learning rate schedule (default: true)")
    
    # System parameters
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    # Output
    parser.add_argument("--experiment_name", type=str, default="baseline",
                       help="Experiment name for logging")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="bnn-training",
                       help="W&B project name")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to experiment directory to resume from (e.g., experiments/baseline_cifar10_20251025_153141)")
    
    # Architecture-specific
    parser.add_argument("--mlp_hidden_sizes", type=int, nargs='+', default=[128, 64],
                       help="Hidden layer sizes for MLP")
    parser.add_argument("--cnn_conv_channels", type=int, nargs='+', default=[32, 64],
                       help="Conv channel sizes for CNN")
    parser.add_argument("--cnn_fc_hidden_sizes", type=int, nargs='+', default=[512],
                       help="FC layer sizes for CNN")
    parser.add_argument("--cnn_conv_layers_per_block", type=int, default=2,
                       help="Number of conv layers per block in CNN")
    parser.add_argument("--cnn_kernel_size", type=int, default=3,
                       help="Kernel size for CNN convolutions")
    parser.add_argument("--cnn_pool_size", type=int, default=2,
                       help="Pool size for CNN max pooling")
    
    args = parser.parse_args()
    
    # Create config
    config = vars(args)
    
    # Create experiment directory
    exp_dir = create_experiment_dir("../experiments", config['experiment_name'])
    save_config(config, exp_dir)
    
    # Train
    train_baseline(config, exp_dir, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
