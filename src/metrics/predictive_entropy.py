"""Predictive entropy metric for BNN uncertainty quantification.

Implementation adapted from torch-uncertainty:
https://github.com/torch-uncertainty/torch-uncertainty/blob/main/src/torch_uncertainty/metrics/classification/mutual_information.py
"""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("predictive_entropy")
class PredictiveEntropy(BaseMetric):
    """Computes predictive entropy (total uncertainty) from BNN ensemble.
    
    Measures the entropy of the averaged predictive distribution (ensemble probabilities).
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly.
        
            Uses all_preds (BNN ensemble) if available, falls back to y_pred (single forward pass).
        """
        output = engine.state.output
        if 'all_preds' in output:
            all_preds = output['all_preds']
        else:
            all_preds = output['y_pred'].unsqueeze(0)
        probs = torch.softmax(all_preds, dim=2) 
        
        # Ensemble probabilities 
        ens_probs = probs.mean(dim=0)
        
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)
        
        self._sum += entropy_mean.sum().item()
        self._count += len(entropy_mean)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
