"""Categorical likelihood for BNN classification."""

import torch.nn.functional as F
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("categorical")
class CategoricalLikelihood:
    """Log-likelihood for categorical classification with logits."""
    
    def __call__(self, y_pred, y):
        """Compute log-likelihood.
        
        Args:
            y_pred: Model predictions (logits), shape [B, C]
            y: Ground truth labels, shape [B]
        
        Returns:
            Log-likelihood (scalar tensor)
        """
        log_probs = F.log_softmax(y_pred, dim=1)
        log_likelihood = log_probs[range(len(y)), y].mean()
        return log_likelihood

