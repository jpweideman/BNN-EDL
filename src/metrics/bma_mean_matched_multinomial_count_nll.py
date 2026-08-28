"""BMA mean-matched multinomial count NLL metric for eBNN ensemble evaluation.

The control likelihood of the count experiment: it keeps each sample's mean
prediction m = alpha / alpha_0 and scores the counts under Mult(R, m). This
removes only the finite total concentration from the predictive likelihood.
"""

import math
import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_mean_matched_multinomial_count_nll")
class BMAMeanMatchedMultinomialCountNLL(BaseMetric):
    """BMA count NLL: log-mean-exp of per-sample Mult(R, alpha / alpha_0) log probabilities."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        alpha = output['all_preds']
        counts = output['y']

        means = alpha / alpha.sum(dim=-1, keepdim=True)
        log_coeff = torch.lgamma(counts.sum(dim=-1) + 1) - torch.lgamma(counts + 1).sum(dim=-1)
        sample_log_probs = log_coeff + torch.xlogy(counts, means).sum(dim=-1)
        log_bma = torch.logsumexp(sample_log_probs, dim=0) - math.log(len(alpha))

        self._sum += -log_bma.sum().item()
        self._count += len(counts)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
