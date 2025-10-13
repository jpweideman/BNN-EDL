"""
Utility functions for BNN models.
"""

import torch
import numpy as np


def set_random_seeds(seed: int = 42):
    """
    Random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

