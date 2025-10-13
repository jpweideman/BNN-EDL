"""
Evidential Deep Learning (EDL) loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def relu_evidence(y: torch.Tensor) -> torch.Tensor:
    """
    Convert network output to evidence using ReLU activation.
    
    Args:
        y: Network output logits
        
    Returns:
        Evidence (alpha - 1)
    """
    return F.relu(y)


def exp_evidence(y: torch.Tensor) -> torch.Tensor:
    """
    Convert network output to evidence using exponential activation.
    
    Args:
        y: Network output logits
        
    Returns:
        Evidence (alpha - 1)
    """
    return torch.exp(torch.clamp(y, max=10))  # Clamp to prevent overflow


def softplus_evidence(y: torch.Tensor) -> torch.Tensor:
    """
    Convert network output to evidence using softplus activation.
    
    Args:
        y: Network output logits
        
    Returns:
        Evidence (alpha - 1)
    """
    return F.softplus(y)


def edl_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    num_classes: int,
    annealing_step: int = 10,
    evidence_func: str = "relu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Evidential Deep Learning loss.
    
    Args:
        outputs: Network outputs (logits)
        targets: True class labels
        epoch: Current training epoch
        num_classes: Number of classes
        annealing_step: Number of epochs for KL annealing
        evidence_func: Evidence function to use ('relu', 'exp', 'softplus')
        
    Returns:
        Tuple of (total_loss, classification_loss, kl_loss)
    """
    # Convert outputs to evidence
    if evidence_func == "relu":
        evidence = relu_evidence(outputs)
    elif evidence_func == "exp":
        evidence = exp_evidence(outputs)
    elif evidence_func == "softplus":
        evidence = softplus_evidence(outputs)
    else:
        raise ValueError(f"Unknown evidence function: {evidence_func}")
    
    # Dirichlet parameters (alpha)
    alpha = evidence + 1
    
    # Strength of evidence
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Classification loss (negative log likelihood)
    # Convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes).float()
    
    # Expected probability under Dirichlet
    prob = alpha / S
    
    # Classification loss
    classification_loss = torch.sum(targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    classification_loss = torch.mean(classification_loss)
    
    # KL divergence loss (regularization towards uniform Dirichlet)
    kl_alpha = (alpha - 1) * (1 - targets_one_hot) + 1
    kl_div = kl_divergence(kl_alpha, num_classes)
    
    # Annealing coefficient
    annealing_coef = torch.min(torch.tensor(1.0), torch.tensor(epoch / annealing_step))
    
    # Total loss
    kl_loss = annealing_coef * kl_div
    total_loss = classification_loss + kl_loss
    
    return total_loss, classification_loss, kl_loss


def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute KL divergence between Dirichlet distribution and uniform prior.
    
    Args:
        alpha: Dirichlet parameters
        num_classes: Number of classes
        
    Returns:
        KL divergence
    """
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    
    second_term = (
        (alpha - ones)
        * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)
    
    kl = first_term + second_term
    return torch.mean(kl)


def edl_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    epoch: int,
    num_classes: int,
    annealing_step: int = 10,
    evidence_func: str = "relu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute EDL loss with MSE-based classification loss.
    
    Args:
        outputs: Network outputs (logits)
        targets: True class labels
        epoch: Current training epoch
        num_classes: Number of classes
        annealing_step: Number of epochs for KL annealing
        evidence_func: Evidence function to use
        
    Returns:
        Tuple of (total_loss, classification_loss, kl_loss)
    """
    # Convert outputs to evidence
    if evidence_func == "relu":
        evidence = relu_evidence(outputs)
    elif evidence_func == "exp":
        evidence = exp_evidence(outputs)
    elif evidence_func == "softplus":
        evidence = softplus_evidence(outputs)
    else:
        raise ValueError(f"Unknown evidence function: {evidence_func}")
    
    # Dirichlet parameters
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # Convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes).float()
    
    # Expected probability
    prob = alpha / S
    
    # MSE loss between expected probabilities and targets
    classification_loss = torch.sum((targets_one_hot - prob) ** 2, dim=1)
    classification_loss = torch.mean(classification_loss)
    
    # KL divergence loss
    kl_alpha = (alpha - 1) * (1 - targets_one_hot) + 1
    kl_div = kl_divergence(kl_alpha, num_classes)
    
    # Annealing
    annealing_coef = torch.min(torch.tensor(1.0), torch.tensor(epoch / annealing_step))
    kl_loss = annealing_coef * kl_div
    
    total_loss = classification_loss + kl_loss
    
    return total_loss, classification_loss, kl_loss


def compute_uncertainty_metrics(alpha: torch.Tensor) -> dict:
    """
    Compute various uncertainty metrics from Dirichlet parameters.
    
    Args:
        alpha: Dirichlet parameters [batch_size, num_classes]
        
    Returns:
        Dictionary with uncertainty metrics
    """
    # Strength of evidence
    S = torch.sum(alpha, dim=1)
    
    # Expected probability
    prob = alpha / S.unsqueeze(1)
    
    # Aleatoric uncertainty (expected entropy)
    aleatoric = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
    
    # Epistemic uncertainty (mutual information)
    digamma_sum = torch.digamma(S + 1)
    digamma_alpha = torch.digamma(alpha + 1)
    epistemic = torch.sum(prob * (digamma_sum.unsqueeze(1) - digamma_alpha), dim=1)
    
    # Total uncertainty
    total = aleatoric + epistemic
    
    # Confidence (max probability)
    confidence = torch.max(prob, dim=1)[0]
    
    return {
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': total,
        'confidence': confidence,
        'strength': S
    }
