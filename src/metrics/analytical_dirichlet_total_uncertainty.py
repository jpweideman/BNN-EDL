"""Analytical total uncertainty for Dirichlet outputs."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("analytical_dirichlet_total_uncertainty")
class AnalyticalDirichletTotalUncertainty(BaseMetric):
    """Computes total uncertainty from the analytical Dirichlet decomposition.

    Mean entropy of the predicted class probabilities, averaged over posterior samples.
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' in output:
            all_preds = output['all_preds']
        else:
            all_preds = output['y_pred'].unsqueeze(0)

        S = all_preds.sum(dim=-1, keepdim=True)
        probs = all_preds / S
        total_uncertainty = torch.special.entr(probs).sum(dim=-1).mean(dim=0)

        self._sum += total_uncertainty.sum().item()
        self._count += len(total_uncertainty)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
