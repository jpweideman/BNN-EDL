"""Negative log-likelihood metric for classification."""

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("nll")
class NLL(Metric):
    """Computes negative log-likelihood for classification predictions."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        y_pred = output['y_pred']
        y = output['y']
        
        log_probs = F.log_softmax(y_pred, dim=-1)
        nll = -log_probs[torch.arange(len(y)), y]
        
        self._sum += nll.sum().item()
        self._count += len(nll)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
