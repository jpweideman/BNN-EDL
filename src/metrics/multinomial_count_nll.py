"""Multinomial count NLL metric for single-model evaluation."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("multinomial_count_nll")
class MultinomialCountNLL(BaseMetric):
    """Computes the multinomial count NLL of the current model's predictions."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        logits = output['y_pred']
        counts = output['y']

        log_probs = torch.log_softmax(logits, dim=-1)
        log_coeff = torch.lgamma(counts.sum(dim=-1) + 1) - torch.lgamma(counts + 1).sum(dim=-1)
        nll = -(log_coeff + (counts * log_probs).sum(dim=-1))

        self._sum += nll.sum().item()
        self._count += len(nll)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
