"""Per-input array collection for offline analysis.

Enabled per evaluation split like any other metric, by adding
`- name: array_dump` to that split's metric list. Holds the arrays of the
most recent evaluation run; the array dump handler
(src/training/handlers/array_dump.py) writes them to disk at completion.
"""

import torch
import numpy as np

from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("array_dump")
class ArrayDump(BaseMetric):
    """Collects per-input arrays for the offline analysis layer.

    Per input (averaged over snapshots where they differ): total concentration
    alpha_0, the aleatoric, distributional and epistemic uncertainty terms, max
    mean probability, predicted class, true label, and position in the
    (unshuffled) loader. Per snapshot: the mean vectors (S x N x C) and alpha_0
    (S x N) in float16, which make the full concentration vectors
    reconstructible as alpha = alpha_0 * m.

    Args:
        output_type: 'dirichlet' (concentration parameters) or 'softmax'
            (logits, e.g. the categorical BNN and the deterministic nets)
        store_snapshots: Keep the per-snapshot arrays (default: True). Set
            False on large diagnostic splits that only need the scalars.
    """

    def __init__(self, output_type="dirichlet", store_snapshots=True):
        if output_type not in ("dirichlet", "softmax"):
            raise ValueError(f"output_type must be 'dirichlet' or 'softmax', got '{output_type}'")
        self.output_type = output_type
        self.store_snapshots = store_snapshots
        super().__init__()

    def reset(self):
        self._rows = {
            'alpha0': [], 'aleatoric': [], 'distributional': [],
            'epistemic': [], 'max_mean_prob': [], 'pred_class': [],
            'true_label': [], 'index': [],
        }
        self._snapshot_means = []   # per batch: (S, B, C)
        self._snapshot_alpha0 = []  # per batch: (S, B)
        self._count = 0

    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' in output:
            all_preds = output['all_preds']            # (S, B, C)
        else:
            all_preds = output['y_pred'].unsqueeze(0)  # (1, B, C)
        batch_size = all_preds.shape[1]

        if self.output_type == "dirichlet":
            alpha0_s = all_preds.sum(dim=-1)                            # (S, B)
            m_s = all_preds / alpha0_s.unsqueeze(-1)                    # (S, B, C)
            entropy_s = torch.special.entr(m_s).sum(dim=-1)             # (S, B)
            # Closed forms, matching the analytical_dirichlet_* metrics.
            aleatoric_s = -torch.sum(
                m_s * (torch.digamma(all_preds + 1) - torch.digamma(alpha0_s.unsqueeze(-1) + 1)),
                dim=-1)                                                 # (S, B)
            distributional_s = entropy_s - aleatoric_s                  # (S, B)
        else:  # softmax logits: no distributional term and no alpha_0
            m_s = torch.softmax(all_preds, dim=-1)
            entropy_s = torch.special.entr(m_s).sum(dim=-1)
            alpha0_s = torch.full_like(entropy_s, float('nan'))
            aleatoric_s, distributional_s = entropy_s, torch.zeros_like(entropy_s)

        # Epistemic term: mutual information across snapshots, from the mean
        # vectors, matching bma(_dirichlet)_mutual_information.
        ens = m_s.mean(dim=0)                                           # (B, C)
        entropy_of_mean = torch.special.entr(ens).sum(dim=-1)           # (B,)
        epistemic = torch.clamp(entropy_of_mean - entropy_s.mean(dim=0), min=0)
        max_mean_prob, pred_class = ens.max(dim=-1)

        rows = self._rows
        rows['alpha0'].append(alpha0_s.mean(dim=0).cpu())
        rows['aleatoric'].append(aleatoric_s.mean(dim=0).cpu())
        rows['distributional'].append(distributional_s.mean(dim=0).cpu())
        rows['epistemic'].append(epistemic.cpu())
        rows['max_mean_prob'].append(max_mean_prob.cpu())
        rows['pred_class'].append(pred_class.cpu())
        rows['true_label'].append(output['y'].cpu())
        rows['index'].append(torch.arange(self._count, self._count + batch_size))
        if self.store_snapshots:
            self._snapshot_means.append(m_s.cpu().to(torch.float16))
            self._snapshot_alpha0.append(alpha0_s.cpu().to(torch.float16))
        self._count += batch_size

    @property
    def has_data(self):
        return self._count > 0

    @property
    def data(self):
        """The accumulated arrays as a dict of numpy arrays."""
        arrays = {k: torch.cat(v).numpy() for k, v in self._rows.items()}
        if self.store_snapshots:
            arrays['snapshot_means'] = torch.cat(self._snapshot_means, dim=1).numpy()
            arrays['snapshot_alpha0'] = torch.cat(self._snapshot_alpha0, dim=1).numpy()
        arrays['output_type'] = np.array(self.output_type)
        return arrays

    def compute(self):
        """Number of collected inputs, as a sanity check in the metrics log."""
        return float(self._count)
