"""Categorical likelihood for classification."""

import torch.nn.functional as F
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("categorical")
class CategoricalLikelihood:
    """Categorical likelihood: mean log probability of correct class."""
    
    def __call__(self, y_pred, y):
        log_probs = F.log_softmax(y_pred, dim=1)
        log_likelihood = log_probs[range(len(y)), y].mean()
        return log_likelihood

