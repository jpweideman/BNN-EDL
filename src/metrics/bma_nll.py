"""BMA NLL metric for BNN ensemble evaluation."""

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from src.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("bma_nll")
@METRIC_REGISTRY.register("bma_cross_entropy")
class BMANLL(Metric):
    """Computes negative log-likelihood using Bayesian Model Averaging across samples."""
    
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
        
        probs = torch.softmax(all_preds, dim=2)
        probs_bma = probs.mean(dim=0)
        log_probs = torch.log(probs_bma)
        ce = F.nll_loss(log_probs, y, reduction='sum')
        
        self._sum += ce.item()
        self._count += len(y)
    
    def compute(self):
        if self._count == 0:
            return 0.0
        return self._sum / self._count

