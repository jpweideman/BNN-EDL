"""Cross-entropy loss for classification."""

import torch.nn as nn
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.
    """
    
    def __init__(self):
        """Initialize cross-entropy loss."""
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(self, y_pred, y_true):
        """
        Compute loss.
        
        Args:
            y_pred: Model predictions (logits), shape (batch_size, num_classes)
            y_true: Ground truth labels, shape (batch_size,)
        
        Returns:
            Loss value (scalar tensor)
        """
        return self.loss_fn(y_pred, y_true)

