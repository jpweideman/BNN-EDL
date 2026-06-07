"""Predictive variance metric for BNN uncertainty quantification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_predictive_variance")
class BMAPredictiveVariance(BaseMetric):
    """Computes predictive variance from BNN ensemble.
    
    Measures the spread of predicted probabilities across samples.
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds'] 
        
        # Probabilities from each sample
        probs = torch.softmax(all_preds, dim=2) 
        
        # Variance across samples for each class
        probs_var = probs.var(dim=0)  
        
        # Total variance (sum across classes)
        total_var = probs_var.sum(dim=1)  
        
        self._sum += total_var.sum().item()
        self._count += len(total_var)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
