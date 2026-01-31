"""Evidential classification loss for EDL models."""

from edl_pytorch import evidential_classification as edl_loss
from src.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("evidential_classification")
class EvidentialClassification:
    """Evidential classification loss for EDL models.
        
    Args:
        lamb: Regularization coefficient (default: 0.001)
    """
    
    def __init__(self, lamb=0.001):
        self.lamb = lamb
    
    def __call__(self, alpha, y_true):
        """Compute evidential loss.
        
        Args:
            alpha: Dirichlet parameters [B, C]
            y_true: True labels [B]
            
        Returns:
            Loss value (scalar tensor)
        """
        return edl_loss(alpha, y_true, lamb=self.lamb)
