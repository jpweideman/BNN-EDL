"""Dirichlet strength metric for EDL uncertainty quantification."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_strength")
class DirichletStrength(Metric):
    """Computes average Dirichlet strength from EDL outputs.
    
    Measures the total evidence S = sum(alpha) for each prediction.
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        alpha = output['y_pred']
        
        S = alpha.sum(dim=-1)
        
        self._sum += S.sum().item()
        self._count += len(S)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
