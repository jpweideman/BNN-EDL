from src.registry import OPTIMIZER_REGISTRY
import torch.optim as optim


@OPTIMIZER_REGISTRY.register("adamw")
class AdamWOptimizer:
    """AdamW optimizer wrapper."""
    
    def __init__(self, params, lr, weight_decay=0.01, **kwargs):
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def __getattr__(self, name):
        return getattr(self.optimizer, name)

