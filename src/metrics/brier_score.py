"""Brier score metric for classification."""

import torch.nn.functional as F
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("brier_score")
class BrierScore(BaseMetric):
    """Computes Brier score for classification predictions."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        y_pred = output['y_pred']
        y = output['y']
        
        num_classes = y_pred.shape[-1]
        probs = F.softmax(y_pred, dim=-1)
        y_one_hot = F.one_hot(y, num_classes).float()
        
        brier = (probs - y_one_hot).pow(2).sum(dim=-1)
        
        self._sum += brier.sum().item()
        self._count += len(brier)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
