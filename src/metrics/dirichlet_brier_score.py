"""Brier score metric for Dirichlet-parameterized classification."""

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_brier_score")
class DirichletBrierScore(Metric):
    """Computes Brier score for Dirichlet predictions."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        alpha = output['y_pred']
        y = output['y']
        
        num_classes = alpha.shape[-1]
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        y_one_hot = F.one_hot(y, num_classes).float()
        
        brier = (probs - y_one_hot).pow(2).sum(dim=-1)
        
        self._sum += brier.sum().item()
        self._count += len(brier)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
