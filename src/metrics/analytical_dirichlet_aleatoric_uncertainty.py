"""Analytical aleatoric uncertainty for Dirichlet outputs."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("analytical_dirichlet_aleatoric_uncertainty")
class AnalyticalDirichletAleatoricUncertainty(BaseMetric):
    """Computes aleatoric (data) uncertainty from the analytical Dirichlet decomposition.

    Data uncertainty from the shape of the evidence, averaged over posterior samples.
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
        aleatoric_uncertainty = -torch.sum(probs * (torch.digamma(all_preds + 1) - torch.digamma(S + 1)), dim=-1).mean(dim=0)

        self._sum += aleatoric_uncertainty.sum().item()
        self._count += len(aleatoric_uncertainty)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
