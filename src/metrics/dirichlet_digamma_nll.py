"""Digamma NLL metric for Dirichlet-parameterized classification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_digamma_nll")
class DirichletDigammaNLL(BaseMetric):
    """Computes digamma expected log-loss for Dirichlet predictions."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        alpha = output['y_pred']
        y = output['y']

        S = alpha.sum(dim=-1)
        nll = torch.digamma(S) - torch.digamma(alpha[torch.arange(len(y)), y])

        self._sum += nll.sum().item()
        self._count += len(nll)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
