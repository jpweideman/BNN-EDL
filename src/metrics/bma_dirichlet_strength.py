"""BMA Dirichlet strength metric for Dirichlet BNN ensemble."""

from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_strength")
class BMADirichletStrength(BaseMetric):
    """Computes average Dirichlet strength using BMA for Dirichlet BNN ensemble."""

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        output = engine.state.output
        if 'all_preds' not in output:
            return

        S = output['all_preds'].sum(dim=-1).mean(dim=0)

        self._sum += S.sum().item()
        self._count += len(S)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
