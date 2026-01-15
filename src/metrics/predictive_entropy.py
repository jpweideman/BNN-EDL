"""Predictive entropy metric for BNN uncertainty."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("predictive_entropy")
class PredictiveEntropy(Metric):
    """Computes predictive entropy: H[p(y|x,D)] = -sum p(y|x,D) log p(y|x,D)"""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        
        # Average probabilities across samples
        probs = torch.softmax(all_preds, dim=-1).mean(dim=0)
        
        # Compute entropy per example
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        self._sum += entropy.sum().item()
        self._count += len(entropy)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count


