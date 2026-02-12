"""Negative log-likelihood metric for Dirichlet-parameterized classification."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_nll")
class DirichletNLL(Metric):
    """Computes negative log-likelihood for Dirichlet predictions."""
    
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
        
        S = alpha.sum(dim=-1)
        alpha_y = alpha[torch.arange(len(y)), y]
        nll = -torch.log(alpha_y / S)
        
        self._sum += nll.sum().item()
        self._count += len(nll)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
