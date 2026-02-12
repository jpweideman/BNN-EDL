"""BMA negative log-likelihood metric for Dirichlet BNN ensemble."""

import torch
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_dirichlet_nll")
class BMADirichletNLL(Metric):
    """Computes negative log-likelihood using BMA for Dirichlet BNN ensemble."""
    
    def reset(self):
        self._sum = 0.0
        self._count = 0
    
    def update(self, output):
        pass
    
    def iteration_completed(self, engine):
        """Override to access engine.state.output directly."""
        output = engine.state.output
        all_preds = output['all_preds']
        y = output['y']
        
        S = all_preds.sum(dim=-1)
        probs = all_preds / S.unsqueeze(-1)
        probs_bma = probs.mean(dim=0)
        
        nll = -torch.log(probs_bma[torch.arange(len(y)), y])
        
        self._sum += nll.sum().item()
        self._count += len(nll)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count
