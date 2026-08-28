"""Dirichlet-multinomial count NLL metric for single-model evaluation."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_multinomial_count_nll")
class DirichletMultinomialCountNLL(BaseMetric):
    """Computes the Dirichlet-multinomial count NLL of the current model's predictions."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        alpha = output['y_pred']
        counts = output['y']

        alpha0 = alpha.sum(dim=-1)
        totals = counts.sum(dim=-1)
        log_coeff = torch.lgamma(totals + 1) - torch.lgamma(counts + 1).sum(dim=-1)
        nll = -(log_coeff + torch.lgamma(alpha0) - torch.lgamma(totals + alpha0)
                + (torch.lgamma(counts + alpha) - torch.lgamma(alpha)).sum(dim=-1))

        self._sum += nll.sum().item()
        self._count += len(nll)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
