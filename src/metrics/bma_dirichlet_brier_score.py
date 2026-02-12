"""BMA Brier score metric for Dirichlet BNN ensemble."""

import torch.nn.functional as F
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_brier_score")
class BMADirichletBrierScore(BaseMetric):
    """Computes Brier score using BMA for Dirichlet BNN ensemble."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        y = output['y']
        
        num_classes = all_preds.shape[-1]
        S = all_preds.sum(dim=-1, keepdim=True)
        probs = (all_preds / S).mean(dim=0)
        y_one_hot = F.one_hot(y, num_classes).float()
        
        brier = (probs - y_one_hot).pow(2).sum(dim=-1)
        
        self._sum += brier.sum().item()
        self._count += len(brier)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
