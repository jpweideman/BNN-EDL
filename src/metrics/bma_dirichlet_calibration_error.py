"""BMA calibration error metric for Dirichlet BNN ensemble."""

import torch
from ignite.metrics import Metric
from torchmetrics.classification import MulticlassCalibrationError
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_calibration_error")
class BMADirichletCalibrationError(Metric):
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
        super().__init__()
    
    def reset(self):
        self.torchmetric.reset()
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        all_preds = output['all_preds']
        y = output['y']
        
        S = all_preds.sum(dim=-1, keepdim=True)
        probs = (all_preds / S).mean(dim=0)
        
        self.torchmetric.update(probs, y)
    
    def compute(self):
        result = self.torchmetric.compute()
        return result.item() if torch.is_tensor(result) else result
