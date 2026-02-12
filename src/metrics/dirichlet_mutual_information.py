"""Mutual information metric for Dirichlet BNN uncertainty quantification."""

import torch
from src.metrics.base import BaseMetric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("dirichlet_mutual_information")
class DirichletMutualInformation(BaseMetric):
    """Computes mutual information (epistemic uncertainty) from Dirichlet BNN ensemble.
    
    Measures the disagreement between different models in the ensemble.
    Computed as: MI = Entropy of Ensemble - Mean of Entropies
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        if 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']
        
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
