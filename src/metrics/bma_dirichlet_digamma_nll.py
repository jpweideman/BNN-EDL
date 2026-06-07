"""BMA digamma NLL metric for Dirichlet BNN ensemble."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_digamma_nll")
class BMADirichletDigammaNLL(BaseMetric):
    """Computes digamma expected log-loss using BMA for Dirichlet BNN ensemble."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        if 'all_preds' not in output:
            return

        all_preds = output['all_preds']
        y = output['y']

        S = all_preds.sum(dim=-1)
        alpha_y = all_preds[:, torch.arange(len(y)), y]
        nll = (torch.digamma(S) - torch.digamma(alpha_y)).mean(dim=0)

        self._sum += nll.sum().item()
        self._count += len(nll)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
