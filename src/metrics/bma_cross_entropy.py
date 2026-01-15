"""BMA cross-entropy loss metric for BNN ensemble evaluation."""

import torch.nn.functional as F
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_cross_entropy")
class BMACrossEntropy(Metric):
    """Computes cross-entropy loss using Bayesian Model Averaging (BMA) across samples."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        y = output['y']
        
        # Compute BMA prediction
        y_pred_bma = all_preds.mean(dim=0)
        
        # Compute cross-entropy loss from BMA prediction
        loss = F.cross_entropy(y_pred_bma, y, reduction='sum')
        
        self._sum += loss.item()
        self._count += len(y)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count

