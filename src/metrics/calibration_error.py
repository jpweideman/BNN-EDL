"""Calibration error metric for standard softmax classification.

Uses torchmetrics' MulticlassCalibrationError implementation.
"""

import torch
import torch.nn.functional as F
from src.metrics.base import BaseMetric
from torchmetrics.classification import MulticlassCalibrationError
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("calibration_error")
class CalibrationError(BaseMetric):
    """Computes Expected Calibration Error (ECE) for softmax predictions.

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
        super().__init__()

    def reset(self):
        self.torchmetric.reset()

    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        probs = F.softmax(output['y_pred'], dim=-1)
        self.torchmetric.update(probs, output['y'])

    def compute(self):
        result = self.torchmetric.compute()
        return result.item() if torch.is_tensor(result) else result
