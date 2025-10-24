"""
Utility functions for BNN models.
"""

import math
import torch
import numpy as np


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class CosineLR:
    """
    Cyclical cosine learning rate scheduler for SGLD (cSG-MCMC).
    
    Implements: α_k = (α_0/2) * [cos(π * mod(k-1, ⌈K/M⌉) / ⌈K/M⌉)) + 1]
    
    Where:
    - K is the number of total iterations (epochs)
    - M is the number of cycles
    - β is the fraction of the cycle for which we do optimization
    
    Reference: cSG-MCMC - Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning
    https://arxiv.org/abs/1902.03932
    https://github.com/activatedgeek/understanding-bayesian-classification/blob/main/src/data_aug/optim/lr_scheduler.py
    """
    def __init__(self, init_lr: float, n_cycles: int, n_samples: int, T_max: int, beta: float = 1/4):
        """
        Args:
            init_lr: Initial learning rate
            n_cycles: Number of cycles (M)
            n_samples: Total number of samples to collect after burn-in
            T_max: Total number of iterations/epochs (K)
            beta: Fraction of cycle for optimization (default: 1/4)
        """
        self.init_lr = init_lr
        self.n_cycles = n_cycles
        self.n_samples = n_samples
        self.T_max = T_max
        self.beta = beta
        
        self._cycle_len = int(math.ceil(T_max / n_cycles))
        self._last_beta = 0.
        
        samples_per_cycle = n_samples // n_cycles
        self._thres = ((beta + torch.arange(1, samples_per_cycle + 1) * (1 - beta) / samples_per_cycle) * self._cycle_len).int()
    
    def get_lr(self, epoch: int) -> float:
        """Get learning rate at given epoch (0-indexed)."""
        if epoch >= self.T_max:
            return 0.0
        
        # Position within current cycle (normalized 0 to 1)
        cycle_progress = (epoch % self._cycle_len) / self._cycle_len
        
        # Cosine annealing within cycle
        lr_factor = (math.cos(math.pi * cycle_progress) + 1.0)
        
        # Return scaled learning rate
        return 0.5 * self.init_lr * lr_factor
    
    def should_sample(self, epoch: int) -> bool:
        """
        Determines if we should sample at this epoch.
        Samples are taken at specific optimal points in each cycle.
        
        Args:
            epoch: Current epoch (0-indexed)
            
        Returns:
            True if this epoch should collect a posterior sample
        """
        _t = epoch % self._cycle_len + 1
        return (_t - self._thres).abs().min() == 0

