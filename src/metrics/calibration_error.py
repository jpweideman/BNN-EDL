"""Calibration error metric for BNN ensemble predictions.

Uses torchmetrics' MulticlassCalibrationError implementation.
"""

import torch
from ignite.metrics import Metric
from torchmetrics.classification import MulticlassCalibrationError
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("calibration_error")
class CalibrationError(Metric):
    """Computes Expected Calibration Error (ECE) for BNN ensemble predictions.
    
    Wrapper around torchmetrics' MulticlassCalibrationError that works with
    BNN ensemble outputs.
    
    Args:
        num_classes: Number of classes (required)
        num_bins: Number of bins for confidence (default: 15)
        norm: Norm to use - 'l1' (ECE), 'l2' (RMSCE), or 'max' (MCE) (default: 'l1')
    """
    
    def __init__(self, num_classes, num_bins=15, norm='l1'):
        self.torchmetric = MulticlassCalibrationError(
            num_classes=num_classes,
            n_bins=num_bins,
            norm=norm
        )
        super().__init__()
    
    def reset(self):
        self.torchmetric.reset()
    
    def update(self, output):
        # Ignored, we override iteration_completed
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly (not transformed)."""
        output = engine.state.output
        if not isinstance(output, dict) or 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']  # [S, B, C]
        y = output['y']  # [B]
        
        # BMA probabilities (ensemble average)
        probs = torch.softmax(all_preds, dim=2).mean(dim=0)  # [B, C]
        
        # Update the torchmetrics metric
        self.torchmetric.update(probs, y)
    
    def compute(self):
        result = self.torchmetric.compute()
        return result.item() if torch.is_tensor(result) else result
