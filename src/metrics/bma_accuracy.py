"""BMA accuracy metric for BNN ensemble evaluation."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_accuracy")
class BMAAccuracy(BaseMetric):
    """Computes accuracy using Bayesian Model Averaging (BMA) across samples."""
    
    def reset(self):
        self._correct = 0
        self._total = 0
    
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        y = output['y']
        
        probs = torch.softmax(all_preds, dim=2)
        probs_bma = probs.mean(dim=0)
        predictions = torch.argmax(probs_bma, dim=1)
        
        self._correct += (predictions == y).sum().item()
        self._total += len(y)
    
    def compute(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total

