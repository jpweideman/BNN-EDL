"""BMA multinomial count NLL metric for BNN ensemble evaluation."""

import math
import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_multinomial_count_nll")
class BMAMultinomialCountNLL(BaseMetric):
    """BMA count NLL: log-mean-exp of per-sample multinomial log probabilities."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        all_preds = output['all_preds']
        counts = output['y']

        log_probs = torch.log_softmax(all_preds, dim=-1)
        log_coeff = torch.lgamma(counts.sum(dim=-1) + 1) - torch.lgamma(counts + 1).sum(dim=-1)
        sample_log_probs = log_coeff + (counts * log_probs).sum(dim=-1)
        log_bma = torch.logsumexp(sample_log_probs, dim=0) - math.log(len(all_preds))

        self._sum += -log_bma.sum().item()
        self._count += len(counts)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
