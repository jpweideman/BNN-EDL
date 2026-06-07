"""Expected Brier score metric for Dirichlet-parameterized classification."""

import torch.nn.functional as F
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_expected_brier")
class DirichletExpectedBrier(BaseMetric):
    """Computes expected Brier score for Dirichlet predictions."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        alpha = output['y_pred']
        y = output['y']

        y_one_hot = F.one_hot(y, alpha.shape[-1]).float()
        S = alpha.sum(dim=-1, keepdim=True)
        err = ((y_one_hot - alpha / S) ** 2).sum(dim=-1)
        var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=-1)

        self._sum += (err + var).sum().item()
        self._count += len(err)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
