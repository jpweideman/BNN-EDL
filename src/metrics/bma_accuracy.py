"""BMA accuracy metric for BNN ensemble evaluation."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_accuracy")
class BMAAccuracy(Metric):
    """Computes accuracy using Bayesian Model Averaging (BMA) across samples."""
    
    def reset(self):
        self._correct = 0
        self._total = 0
    
    def update(self, output):
        # Ignored, we override iteration_completed to access engine.state.output directly

        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly (not transformed)."""
        output = engine.state.output
        if not isinstance(output, dict) or 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        y = output['y']
        
        # Compute BMA prediction
        y_pred_bma = all_preds.mean(dim=0)
        predictions = torch.argmax(y_pred_bma, dim=1)
        
        self._correct += (predictions == y).sum().item()
        self._total += len(y)
    
    def compute(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total

