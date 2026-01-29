"""Mutual information metric for BNN uncertainty quantification.

Implementation adapted from torch-uncertainty:
https://github.com/torch-uncertainty/torch-uncertainty/blob/main/src/torch_uncertainty/metrics/classification/mutual_information.py
"""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("mutual_information")
class MutualInformation(Metric):
    """Computes mutual information (epistemic uncertainty) from BNN ensemble.
    
    Measures the disagreement between different models in the ensemble.
    Computed as: MI = Entropy of Ensemble - Mean of Entropies
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        # Ignored, we override iteration_completed to access engine.state.output directly
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly (not transformed)."""
        output = engine.state.output
        if not isinstance(output, dict) or 'all_preds' not in output:
            return
        
        all_preds = output['all_preds']  # [S, B, C]
        
        # Convert to probabilities
        probs = torch.softmax(all_preds, dim=2)  # [S, B, C]
        
        ens_probs = probs.mean(dim=0)  # [B, C]
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)  # [B]
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=0)  # [B]
        
        # Mutual information 
        mutual_info = torch.clamp(entropy_mean - mean_entropy, min=0)  # [B]
        
        self._sum += mutual_info.sum().item()
        self._count += len(mutual_info)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count

