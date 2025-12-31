from src.registry import OPTIMIZER_REGISTRY
import torch.optim as optim


@OPTIMIZER_REGISTRY.register("sgd")
class SGDOptimizer:
    """SGD optimizer wrapper."""
    
    def __init__(self, params, lr, momentum=0.9, weight_decay=0.0, **kwargs):
        self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    def __getattr__(self, name):
        return getattr(self.optimizer, name)

