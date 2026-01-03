import torch.optim.lr_scheduler as lr_scheduler
from src.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("exponential_lr")
class ExponentialLRScheduler:
    def __init__(self, optimizer, gamma):
        self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

