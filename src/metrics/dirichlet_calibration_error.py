"""Calibration error metric for single-sample Dirichlet classification.

Uses torchmetrics' MulticlassCalibrationError implementation.
"""

import torch
from src.metrics.base import BaseMetric
from torchmetrics.classification import MulticlassCalibrationError
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_calibration_error")
class DirichletCalibrationError(BaseMetric):
    """Computes Expected Calibration Error (ECE) for Dirichlet predictions.

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
        alpha = output['y_pred']
        probs = alpha / alpha.sum(dim=-1, keepdim=True)
        self.torchmetric.update(probs, output['y'])

    def compute(self):
        result = self.torchmetric.compute()
        return result.item() if torch.is_tensor(result) else result
