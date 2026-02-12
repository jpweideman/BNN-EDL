"""BMA accuracy metric for Dirichlet BNN ensemble."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_accuracy")
class BMADirichletAccuracy(Metric):
    """Computes accuracy using BMA for Dirichlet BNN ensemble."""
    
    def reset(self):
        self._correct = 0
        self._total = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        all_preds = output['all_preds']
        y = output['y']
        
        S = all_preds.sum(dim=-1, keepdim=True)
        probs = all_preds / S
        probs_bma = probs.mean(dim=0)
        predictions = torch.argmax(probs_bma, dim=1)
        
        self._correct += (predictions == y).sum().item()
        self._total += len(y)
    
    def compute(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total
