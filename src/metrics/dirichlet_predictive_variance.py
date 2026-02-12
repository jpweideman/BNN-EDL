"""Predictive variance metric for Dirichlet BNN uncertainty quantification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_predictive_variance")
class DirichletPredictiveVariance(BaseMetric):
    """Computes predictive variance from Dirichlet BNN ensemble.
    
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
        
        S = all_preds.sum(dim=-1, keepdim=True)
        probs = all_preds / S
        
        probs_var = probs.var(dim=0)
        
        total_var = probs_var.sum(dim=1)
        
        self._sum += total_var.sum().item()
        self._count += len(total_var)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
