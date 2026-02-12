"""Vacuity metric for EDL uncertainty quantification."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("vacuity")
class Vacuity(Metric):
    """Computes vacuity (epistemic uncertainty) from EDL Dirichlet parameters.
    
    Vacuity measures the lack of evidence in the model's predictions.
    Formula: u = K / S, where K is number of classes and S = sum(alpha).
    """
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        alpha = output['y_pred']
        
        K = alpha.shape[-1]
        S = alpha.sum(dim=-1)
        
        vacuity = K / S
        
        self._sum += vacuity.sum().item()
        self._count += len(vacuity)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
