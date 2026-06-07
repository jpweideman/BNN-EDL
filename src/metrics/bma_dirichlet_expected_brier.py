"""BMA expected Brier metric for Dirichlet BNN ensemble."""

import torch.nn.functional as F
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_expected_brier")
class BMADirichletExpectedBrier(BaseMetric):
    """Computes expected Brier score using BMA for Dirichlet BNN ensemble."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        if 'all_preds' not in output:
            return

        all_preds = output['all_preds']
        y = output['y']

        y_one_hot = F.one_hot(y, all_preds.shape[-1]).float()
        S = all_preds.sum(dim=-1, keepdim=True)
        err = ((y_one_hot.unsqueeze(0) - all_preds / S) ** 2).sum(dim=-1)
        var = (all_preds * (S - all_preds) / (S * S * (S + 1))).sum(dim=-1)
        brier = (err + var).mean(dim=0)

        self._sum += brier.sum().item()
        self._count += len(brier)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
