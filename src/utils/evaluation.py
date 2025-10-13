"""
Evaluation utilities for BNN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import confusion_matrix
import warnings


def analyze_predictions(
    bnn,
    dataloader,
    num_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Analyze BNN predictions and uncertainties.
    
    Args:
        bnn: Trained BNN model
        dataloader: Data loader for analysis
        num_samples: Number of posterior samples
        device: Device to use for computation
        
    Returns:
        Dictionary with prediction analysis results
    """
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    all_correct = []
    all_images = []
    
    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # Get predictions and uncertainties
        predictions, uncertainties = bnn.predict_batch(images, num_samples)
        
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1)
        correct = (pred_classes == labels).float()
        
        all_predictions.append(predictions.cpu())
        all_uncertainties.append(uncertainties.cpu())
        all_labels.append(labels.cpu())
        all_correct.append(correct.cpu())
        all_images.append(images.cpu())
    
    return {
        'predictions': torch.cat(all_predictions, dim=0),
        'uncertainties': torch.cat(all_uncertainties, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'correct': torch.cat(all_correct, dim=0),
        'images': torch.cat(all_images, dim=0)
    }


def plot_uncertainty_analysis(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    correct: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot uncertainty analysis visualizations.
    
    Args:
        predictions: Model predictions [N, num_classes]
        uncertainties: Epistemic uncertainties [N]
        correct: Correctness indicators [N]
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Uncertainty distribution for correct vs incorrect
    correct_uncertainties = uncertainties[correct == 1]
    incorrect_uncertainties = uncertainties[correct == 0]
    
    axes[0, 0].hist(correct_uncertainties.numpy(), bins=50, alpha=0.7, 
                   label='Correct', density=True, color='green')
    axes[0, 0].hist(incorrect_uncertainties.numpy(), bins=50, alpha=0.7, 
                   label='Incorrect', density=True, color='red')
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Uncertainty Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy vs uncertainty bins
    n_bins = 10
    uncertainty_sorted, indices = torch.sort(uncertainties)
    bin_size = len(uncertainty_sorted) // n_bins
    
    bin_accuracies = []
    bin_uncertainties = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(uncertainty_sorted)
        
        bin_indices = indices[start_idx:end_idx]
        bin_accuracy = correct[bin_indices].mean().item()
        bin_uncertainty = uncertainties[bin_indices].mean().item()
        
        bin_accuracies.append(bin_accuracy)
        bin_uncertainties.append(bin_uncertainty)
    
    axes[0, 1].plot(bin_uncertainties, bin_accuracies, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Average Uncertainty (bin)')
    axes[0, 1].set_ylabel('Accuracy (bin)')
    axes[0, 1].set_title('Accuracy vs Uncertainty')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confidence vs accuracy (reliability diagram)
    confidences = torch.max(predictions, dim=1)[0]
    conf_sorted, conf_indices = torch.sort(confidences)
    
    conf_bin_accuracies = []
    conf_bin_confidences = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(conf_sorted)
        
        bin_indices = conf_indices[start_idx:end_idx]
        bin_accuracy = correct[bin_indices].mean().item()
        bin_confidence = confidences[bin_indices].mean().item()
        
        conf_bin_accuracies.append(bin_accuracy)
        conf_bin_confidences.append(bin_confidence)
    
    axes[1, 0].plot(conf_bin_confidences, conf_bin_accuracies, 'o-', 
                   linewidth=2, markersize=8, label='BNN')
    axes[1, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Reliability Diagram')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty vs confidence scatter
    axes[1, 1].scatter(confidences.numpy(), uncertainties.numpy(), 
                      c=correct.numpy(), cmap='RdYlGn', alpha=0.6)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Epistemic Uncertainty')
    axes[1, 1].set_title('Confidence vs Uncertainty')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Correct (1) / Incorrect (0)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_examples(
    images: torch.Tensor,
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    labels: torch.Tensor,
    n_examples: int = 16,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
):
    """
    Plot example predictions with uncertainties.
    
    Args:
        images: Input images [N, C, H, W] or [N, H, W]
        predictions: Model predictions [N, num_classes]
        uncertainties: Epistemic uncertainties [N]
        labels: True labels [N]
        n_examples: Number of examples to show
        figsize: Figure size
        save_path: Path to save the plot
    """
    pred_classes = torch.argmax(predictions, dim=1)
    max_probs = torch.max(predictions, dim=1)[0]
    
    # Handle different image shapes
    if len(images.shape) == 4:  # [N, C, H, W]
        if images.shape[1] == 1:  # Grayscale
            images_viz = images.squeeze(1)  # Remove channel dimension
        else:  # RGB
            images_viz = images.permute(0, 2, 3, 1)  # [N, H, W, C]
    else:  # Already [N, H, W] or flattened
        if len(images.shape) == 2:  # Flattened
            # Assume square images (like MNIST 784 -> 28x28)
            img_size = int(np.sqrt(images.shape[1]))
            images_viz = images.view(-1, img_size, img_size)
        else:
            images_viz = images
    
    # Select examples
    n_examples = min(n_examples, len(images))
    indices = torch.randperm(len(images))[:n_examples]
    
    # Create subplot grid
    grid_size = int(np.ceil(np.sqrt(n_examples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for i in range(grid_size * grid_size):
        if i < n_examples:
            idx = indices[i]
            
            # Display image
            if len(images_viz.shape) == 4:  # RGB
                axes[i].imshow(images_viz[idx])
            else:  # Grayscale
                axes[i].imshow(images_viz[idx], cmap='gray')
            
            # Create title with prediction info
            true_label = labels[idx].item()
            pred_label = pred_classes[idx].item()
            confidence = max_probs[idx].item()
            uncertainty = uncertainties[idx].item()
            
            # Color: green if correct, red if incorrect
            color = 'green' if true_label == pred_label else 'red'
            
            title = f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}, Unc: {uncertainty:.3f}'
            axes[i].set_title(title, fontsize=8, color=color)
            axes[i].axis('off')
        else:
            axes[i].axis('off')
    
    plt.suptitle('BNN Predictions with Uncertainty\n(Green=Correct, Red=Incorrect)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for predictions.
    
    Args:
        predictions: Model predictions [N, num_classes]
        labels: True labels [N]
        class_names: Names of classes (optional)
        save_path: Path to save the plot
    """
    pred_classes = torch.argmax(predictions, dim=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels.numpy(), pred_classes.numpy())
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_calibration_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics.
    
    Args:
        predictions: Model predictions [N, num_classes]
        labels: True labels [N]
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    confidences = torch.max(predictions, dim=1)[0]
    pred_classes = torch.argmax(predictions, dim=1)
    accuracies = (pred_classes == labels).float()
    
    # ECE (Expected Calibration Error)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            mce = max(mce, calibration_error.item())
    
    return {
        'ece': ece.item(),
        'mce': mce,
        'accuracy': accuracies.mean().item(),
        'avg_confidence': confidences.mean().item()
    }

