"""Mutual information metric for Dirichlet uncertainty quantification.

Works for both standard EDL (single y_pred) and BNN ensembles (all_preds).
For EDL: always 0 (single prediction, no ensemble disagreement).
For BNN: disagreement between ensemble members (epistemic uncertainty).
"""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_mutual_information")
class DirichletMutualInformation(BaseMetric):
    """Computes mutual information (epistemic uncertainty) from Dirichlet output.

    MI = Entropy of ensemble mean - Mean of individual entropies.
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
        ens_probs = probs.mean(dim=0)
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=0)
        mutual_info = torch.clamp(entropy_mean - mean_entropy, min=0)

        self._sum += mutual_info.sum().item()
        self._count += len(mutual_info)

    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
