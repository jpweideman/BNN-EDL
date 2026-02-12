"""Brier score metric for classification."""

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("brier_score")
class BrierScore(Metric):
    """Computes Brier score for classification predictions."""
    
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
        
        num_classes = y_pred.shape[-1]
        probs = F.softmax(y_pred, dim=-1)
        y_one_hot = F.one_hot(y, num_classes).float()
        
        brier = (probs - y_one_hot).pow(2).sum(dim=-1)
        
        self._sum += brier.sum().item()
        self._count += len(brier)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
