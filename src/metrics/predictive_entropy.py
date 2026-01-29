"""Predictive entropy metric for BNN uncertainty quantification.

Implementation adapted from torch-uncertainty:
https://github.com/torch-uncertainty/torch-uncertainty/blob/main/src/torch_uncertainty/metrics/classification/mutual_information.py
"""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("predictive_entropy")
class PredictiveEntropy(Metric):
    """Computes predictive entropy (total uncertainty) from BNN ensemble.
    
    Measures the entropy of the averaged predictive distribution (ensemble probabilities).
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        # Ignored, we override iteration_completed to access engine.state.output directly
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly (not transformed)."""
        output = engine.state.output
        if not isinstance(output, dict) or 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']  # [S, B, C]
        
        # Convert to probabilities
        probs = torch.softmax(all_preds, dim=2)  # [S, B, C]
        
        # Ensemble probabilities 
        ens_probs = probs.mean(dim=0)  # [B, C]
        
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)  # [B]
        
        self._sum += entropy_mean.sum().item()
        self._count += len(entropy_mean)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count

