"""Predictive entropy metric for Dirichlet BNN uncertainty quantification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_predictive_entropy")
class DirichletPredictiveEntropy(BaseMetric):
    """Computes predictive entropy (total uncertainty) from Dirichlet BNN ensemble.
    
    Measures the entropy of the averaged predictive distribution (ensemble probabilities).
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
        
        ens_probs = probs.mean(dim=0)
        
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)
        
        self._sum += entropy_mean.sum().item()
        self._count += len(entropy_mean)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
