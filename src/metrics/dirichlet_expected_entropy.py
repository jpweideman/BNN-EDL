"""Expected data entropy metric for Dirichlet BNN uncertainty quantification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_expected_entropy")
class DirichletExpectedEntropy(BaseMetric):
    """Computes expected data entropy (aleatoric uncertainty) from Dirichlet BNN ensemble.
    
    Measures the average entropy of individual model predictions.
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
        
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=0)
        
        self._sum += mean_entropy.sum().item()
        self._count += len(mean_entropy)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
