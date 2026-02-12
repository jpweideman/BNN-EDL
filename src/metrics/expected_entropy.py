"""Expected data entropy metric for BNN uncertainty quantification.

Implementation adapted from torch-uncertainty:
https://github.com/torch-uncertainty/torch-uncertainty/blob/main/src/torch_uncertainty/metrics/classification/mutual_information.py
"""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("expected_entropy")
class ExpectedEntropy(Metric):
    """Computes expected data entropy (aleatoric uncertainty) from BNN ensemble.
    
    Measures the average entropy of individual model predictions.
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
        
        all_preds = output['all_preds']  # [S, B, C]
        
        # Convert to probabilities
        probs = torch.softmax(all_preds, dim=2)  # [S, B, C]
        
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=0)  # [B]
        
        self._sum += mean_entropy.sum().item()
        self._count += len(mean_entropy)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count


