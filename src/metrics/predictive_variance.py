"""Predictive variance metric for BNN uncertainty."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("predictive_variance")
class PredictiveVariance(Metric):
    """Computes variance of predictions across ensemble samples."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        
        # Variance across samples
        variance = all_preds.var(dim=0).mean(dim=-1)
        
        self._sum += variance.sum().item()
        self._count += len(variance)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count


