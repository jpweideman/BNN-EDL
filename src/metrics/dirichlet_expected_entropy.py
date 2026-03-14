"""Expected data entropy metric for Dirichlet uncertainty quantification.

Works for both standard EDL (single y_pred) and BNN ensembles (all_preds).
For EDL: entropy of the single prediction (equals predictive entropy, MI=0).
For BNN: mean entropy of individual ensemble member predictions (aleatoric).
"""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_expected_entropy")
class DirichletExpectedEntropy(BaseMetric):
    """Computes expected data entropy (aleatoric uncertainty) from Dirichlet output.

    Uses all_preds (BNN ensemble) if available, falls back to y_pred (EDL).
    """

    def reset(self):
        self._sum = 0.0
        self._count = 0

    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' in output:
            all_preds = output['all_preds']
        else:
            all_preds = output['y_pred'].unsqueeze(0)

        S = all_preds.sum(dim=-1, keepdim=True)
        probs = all_preds / S
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=0)

        self._sum += mean_entropy.sum().item()
        self._count += len(mean_entropy)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
