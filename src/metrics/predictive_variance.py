"""Predictive variance metric for BNN uncertainty quantification."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("predictive_variance")
class PredictiveVariance(Metric):
    """Computes predictive variance from BNN ensemble.
    
    Measures the spread of predicted probabilities across samples.
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
        
        all_preds = output['all_preds']  # [S, B, C]
        
        # Probabilities from each sample
        probs = torch.softmax(all_preds, dim=2)  # [S, B, C]
        
        # Variance across samples for each class
        probs_var = probs.var(dim=0)  # [B, C]
        
        # Total variance (sum across classes)
        total_var = probs_var.sum(dim=1)  # [B]
        
        self._sum += total_var.sum().item()
        self._count += len(total_var)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
