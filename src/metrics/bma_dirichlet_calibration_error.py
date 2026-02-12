"""BMA calibration error metric for Dirichlet BNN ensemble."""

import torch
from src.metrics.base import BaseMetric
from torchmetrics.classification import MulticlassCalibrationError
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_calibration_error")
class BMADirichletCalibrationError(BaseMetric):
    """Computes Expected Calibration Error (ECE) for Dirichlet BNN ensemble.
    
    Args:
        num_classes: Number of classes
        num_bins: Number of bins for confidence (default: 15)
        norm: Norm to use - 'l1' (ECE), 'l2' (RMSCE), or 'max' (MCE) (default: 'l1')
    """
    
    def __init__(self, num_classes, num_bins=15, norm='l1'):
        self.torchmetric = MulticlassCalibrationError(
            num_classes=num_classes,
            n_bins=num_bins,
            norm=norm
        )
        self._has_data = False
        super().__init__()
    
    def reset(self):
        self.torchmetric.reset()
        self._has_data = False
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        y = output['y']
        
        S = all_preds.sum(dim=-1, keepdim=True)
        probs = (all_preds / S).mean(dim=0)
        
        self.torchmetric.update(probs, y)
        self._has_data = True
    
    def compute(self):
        if not self._has_data:
            return 0.0
        result = self.torchmetric.compute()
        return result.item() if torch.is_tensor(result) else result
