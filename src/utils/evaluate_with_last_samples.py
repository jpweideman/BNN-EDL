#!/usr/bin/env python3
"""
Evaluate a completed BNN experiment using only the last N posterior samples.

Usage:
    python evaluate_with_last_samples.py <run_name> --last_n_samples 20
    python evaluate_with_last_samples.py <run_name> -n 50 --device cuda
    python evaluate_with_last_samples.py <run_name> -n 100 --save_results results.json
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'training'))

from models.standard_bnn import StandardBNN
from models.base_bnn import create_mlp, create_cnn, create_resnet20, create_resnet18
from datasets.classification import get_mnist_dataloaders, get_cifar10_dataloaders


def find_experiment_dir(run_name_or_path: str) -> Path:
    """
    Find experiment directory by run name or return path if it exists.
    
    Args:
        run_name_or_path: Experiment name (substring match) or full path
        
    Returns:
        Path to experiment directory
        
    Raises:
        ValueError: If experiment not found or ambiguous
    """
    # If it's already a valid path, use it
    if os.path.isdir(run_name_or_path):
        return Path(run_name_or_path)
    
    # Search in experiments folder
    experiments_dir = project_root / 'experiments'
    
    # Look for exact match first
    candidate = experiments_dir / run_name_or_path
    if candidate.is_dir():
        return candidate
    
    # Look for substring matches
    matching_dirs = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and run_name_or_path in exp_dir.name:
            matching_dirs.append(exp_dir)
    
    if len(matching_dirs) == 0:
        raise ValueError(
            f"No experiment found matching '{run_name_or_path}'. "
            f"Looked in {experiments_dir}"
        )
    elif len(matching_dirs) > 1:
        # Sort by modification time, return most recent
        matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"Multiple matches found. Using most recent: {matching_dirs[0].name}")
        return matching_dirs[0]
    
    return matching_dirs[0]


def load_experiment(experiment_dir: Path, device: str = 'auto') -> Tuple[StandardBNN, Dict, Dict]:
    """
    Load a completed BNN experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        device: Device to use ('auto', 'cuda', 'cpu')
        
    Returns:
        Tuple of (bnn, config, dataset_info)
        
    Raises:
        ValueError: If required files not found
    """
    experiment_dir = Path(experiment_dir)
    
    # Check required files
    config_path = experiment_dir / 'config.json'
    model_path = experiment_dir / 'model.pt'
    results_path = experiment_dir / 'results.json'
    
    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise ValueError(f"Model checkpoint not found: {model_path}")
    if not results_path.exists():
        raise ValueError(f"Results not found: {results_path}")
    
    # Load config and results
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    dataset_info = results['dataset_info']
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model architecture
    arch_name = config['architecture']
    input_shape = tuple(dataset_info['input_shape'])
    num_classes = dataset_info['num_classes']
    
    if arch_name == "mlp":
        input_size = np.prod(input_shape)
        hidden_sizes = config.get('mlp_hidden_sizes', [128, 64])
        model = create_mlp(input_size, hidden_sizes, num_classes)
    elif arch_name == "cnn":
        conv_channels = config.get('cnn_conv_channels', None)
        conv_layers_per_block = config.get('cnn_conv_layers_per_block', 2)
        fc_hidden_sizes = config.get('cnn_fc_hidden_sizes', None)
        kernel_size = config.get('cnn_kernel_size', 3)
        pool_size = config.get('cnn_pool_size', 2)
        model = create_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            conv_channels=conv_channels,
            conv_layers_per_block=conv_layers_per_block,
            fc_hidden_sizes=fc_hidden_sizes,
            kernel_size=kernel_size,
            pool_size=pool_size
        )
    elif arch_name == "resnet20":
        model = create_resnet20(input_shape, num_classes)
    elif arch_name == "resnet18":
        model = create_resnet18(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    # Create BNN
    bnn = StandardBNN(
        model=model,
        num_classes=num_classes,
        prior_std=config['prior_std'],
        temperature=config['temperature'],
        mcmc_method=config['mcmc_method'],
        device=device
    )
    
    # Load model state and posterior samples
    checkpoint = torch.load(model_path, map_location=device)
    bnn.model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'posterior_samples' not in checkpoint or not checkpoint['posterior_samples']:
        raise ValueError("No posterior samples found in checkpoint!")
    
    # Load posterior samples and move to device
    bnn.posterior_samples = [
        {k: v.to(device) if torch.is_tensor(v) else v 
         for k, v in sample.items()}
        for sample in checkpoint['posterior_samples']
    ]
    
    return bnn, config, dataset_info


def get_test_loader(dataset_name: str, batch_size: int = 128, num_workers: int = 0, 
                    architecture: str = "mlp"):
    """Get test dataloader for a dataset."""
    data_root = project_root / 'data'
    flatten_images = (architecture == "mlp")
    
    if dataset_name == "mnist":
        _, _, test_loader = get_mnist_dataloaders(
            data_dir=str(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=1.0,
            flatten=flatten_images
        )
    elif dataset_name == "cifar10":
        _, _, test_loader = get_cifar10_dataloaders(
            data_dir=str(data_root),
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=1.0
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return test_loader


def evaluate_with_last_n_samples(
    bnn: StandardBNN,
    test_loader,
    n_samples: int,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate BNN using only the last n posterior samples.
    
    This matches the exact computation in StandardBNN.evaluate() but uses
    only the last n samples instead of all samples.
    
    Args:
        bnn: Trained BNN model with posterior_samples
        test_loader: Test data loader
        n_samples: Number of last samples to use
        device: Device to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    from torch import func
    import torch.nn as nn
    from tqdm import tqdm
    
    if not hasattr(bnn, 'posterior_samples') or len(bnn.posterior_samples) == 0:
        raise ValueError("No posterior samples available!")
    
    # Get the last n samples
    total_samples = len(bnn.posterior_samples)
    if n_samples > total_samples:
        print(f"Warning: Requested {n_samples} samples but only {total_samples} available. Using all.")
        samples_to_use = bnn.posterior_samples
    else:
        samples_to_use = bnn.posterior_samples[-n_samples:]
    
    all_predictions = []
    all_labels = []
    all_epistemic = []
    
    sum_loss = 0.0
    sum_total_u = 0.0
    sum_aleatoric_u = 0.0
    sum_epistemic_u = 0.0
    total_N = 0
    
    num_samples_used = len(samples_to_use)
    
    eval_pbar = tqdm(test_loader, desc=f"Evaluating (last {num_samples_used} samples)", leave=False)
    
    eps = 1e-8
    batch_idx = 0
    
    for batch in eval_pbar:
        images, labels = batch
        labels = labels.to(device)
        images = images.to(device)
        B = labels.size(0)
        
        # Flatten images if needed
        if bnn._needs_flattened_input and len(images.shape) > 2:
            images = images.view(images.size(0), -1)
        
        # Compute predictions for each sample
        probs_list = []
        for sample_params in samples_to_use:
            with torch.no_grad():
                logits = func.functional_call(bnn.model, sample_params, images)
                probs_list.append(torch.softmax(logits, dim=1))
        probs = torch.stack(probs_list, dim=0)  # [num_samples, batch_size, num_classes]
        
        # Posterior predictive mean
        expected_probs = probs.mean(dim=0)
        expected_probs = torch.clamp(expected_probs, min=eps)
        expected_probs = expected_probs / expected_probs.sum(dim=1, keepdim=True)
        
        all_predictions.append(expected_probs)
        all_labels.append(labels)
        
        # Loss
        loss_batch = torch.nn.functional.nll_loss(torch.log(expected_probs), labels, reduction='mean')
        
        # Entropy-based uncertainty decomposition
        total_u = -(expected_probs * torch.log(expected_probs + eps)).sum(dim=1)
        ent_per_sample = -(probs * torch.log(probs + eps)).sum(dim=2)
        aleatoric_u = ent_per_sample.mean(dim=0)
        epistemic_u = torch.clamp(total_u - aleatoric_u, min=0)
        all_epistemic.append(epistemic_u)
        
        # Accumulate metrics
        sum_loss += float(loss_batch.item()) * B
        sum_total_u += total_u.sum().item()
        sum_aleatoric_u += aleatoric_u.sum().item()
        sum_epistemic_u += epistemic_u.sum().item()
        total_N += B
        
        batch_idx += 1
        eval_pbar.set_postfix({
            'Batch': f"{batch_idx}/{len(test_loader)}",
            'Samples': num_samples_used
        })
    
    # Compute aggregate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_epistemic = torch.cat(all_epistemic, dim=0)
    
    accuracy = bnn.compute_accuracy(all_predictions, all_labels)
    ece = bnn.compute_ece(all_predictions, all_labels)
    avg_uncertainty = all_epistemic.mean().item()
    
    loss = sum_loss / max(1, total_N)
    total_uncertainty = sum_total_u / max(1, total_N)
    aleatoric_uncertainty = sum_aleatoric_u / max(1, total_N)
    epistemic_uncertainty = sum_epistemic_u / max(1, total_N)
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'avg_uncertainty': avg_uncertainty,
        'num_posterior_samples': num_samples_used,
        'num_test_points': len(all_predictions),
        'loss': loss,
        'total_uncertainty': total_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'epistemic_uncertainty': epistemic_uncertainty,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a completed BNN run using only the last N posterior samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using last 20 samples
  python evaluate_with_last_samples.py cifar10_resnet20_lr_0.01 --last_n_samples 20
  
  # Use short flag
  python evaluate_with_last_samples.py cifar10_resnet20_sghmc -n 50
  
  # Save results
  python evaluate_with_last_samples.py cifar10_resnet20_lr_0.01 -n 100 --save_results results.json
  
  # Use GPU
  python evaluate_with_last_samples.py cifar10_resnet20_lr_0.01 -n 50 --device cuda --batch_size 256
        """
    )
    
    parser.add_argument(
        "run_name",
        help="Experiment run name (substring match or full path)"
    )
    parser.add_argument(
        "-n", "--last_n_samples",
        type=int,
        required=True,
        help="Number of last posterior samples to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0)"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default=None,
        help="Path to save results as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        # Find and load experiment
        print(f"\nSearching for experiment: {args.run_name}")
        exp_dir = find_experiment_dir(args.run_name)
        print(f"Found experiment: {exp_dir.name}")
        
        print(f"Loading experiment...")
        bnn, config, dataset_info = load_experiment(exp_dir, device=args.device)
        
        print(f"\nExperiment info:")
        print(f"  Dataset: {config['dataset'].upper()}")
        print(f"  Architecture: {config['architecture']}")
        print(f"  MCMC Method: {config['mcmc_method'].upper()}")
        print(f"  Total posterior samples: {len(bnn.posterior_samples)}")
        print(f"  Using last: {args.last_n_samples} samples")
        
        # Get test loader
        test_loader = get_test_loader(
            config['dataset'],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            architecture=config['architecture']
        )
        print(f"  Test set size: {len(test_loader.dataset)}")
        
        # Evaluate
        print(f"\nEvaluating with last {args.last_n_samples} samples...")
        metrics = evaluate_with_last_n_samples(
            bnn=bnn,
            test_loader=test_loader,
            n_samples=args.last_n_samples,
            device=bnn.device
        )
        
        # Display results
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:              {metrics['accuracy']:.4f}")
        print(f"ECE:                   {metrics['ece']:.4f}")
        print(f"Loss:                  {metrics['loss']:.4f}")
        print(f"Total Uncertainty:     {metrics['total_uncertainty']:.4f}")
        print(f"Aleatoric Uncertainty: {metrics['aleatoric_uncertainty']:.4f}")
        print(f"Epistemic Uncertainty: {metrics['epistemic_uncertainty']:.4f}")
        print(f"Avg Uncertainty:       {metrics['avg_uncertainty']:.4f}")
        print(f"{'='*60}\n")
        
        # Save if requested
        if args.save_results:
            output_path = Path(args.save_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results_to_save = {
                'run_name': exp_dir.name,
                'last_n_samples': args.last_n_samples,
                'total_posterior_samples': len(bnn.posterior_samples),
                'config': config,
                'metrics': metrics
            }
            
            with open(output_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            print(f"Results saved to: {output_path}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
