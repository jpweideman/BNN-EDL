"""Predictive entropy metric for Dirichlet uncertainty quantification.

Works for both standard EDL (single y_pred) and BNN ensembles (all_preds).
For EDL: entropy of the single Dirichlet mean prediction.
For BNN: entropy of the ensemble-averaged Dirichlet mean prediction.
"""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_predictive_entropy")
class DirichletPredictiveEntropy(BaseMetric):
    """Computes predictive entropy (total uncertainty) from Dirichlet output.

    Uses all_preds (BNN ensemble) if available, falls back to y_pred (EDL).
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
        ens_probs = probs.mean(dim=0)
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)

        self._sum += entropy_mean.sum().item()
        self._count += len(entropy_mean)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
