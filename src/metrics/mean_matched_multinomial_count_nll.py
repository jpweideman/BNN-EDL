"""Mean-matched multinomial count NLL metric for single-model evaluation.

The control likelihood of the count experiment: it keeps the model's mean
prediction m = alpha / alpha_0 and scores the counts under Mult(R, m).
"""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("mean_matched_multinomial_count_nll")
class MeanMatchedMultinomialCountNLL(BaseMetric):
    """Computes the Mult(R, alpha / alpha_0) count NLL of the current model's predictions."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        alpha = output['y_pred']
        counts = output['y']

        means = alpha / alpha.sum(dim=-1, keepdim=True)
        log_coeff = torch.lgamma(counts.sum(dim=-1) + 1) - torch.lgamma(counts + 1).sum(dim=-1)
        nll = -(log_coeff + torch.xlogy(counts, means).sum(dim=-1))

        self._sum += nll.sum().item()
        self._count += len(nll)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
