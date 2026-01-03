import torch.optim.lr_scheduler as lr_scheduler
from src.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("step_lr")
class StepLRScheduler:
    def __init__(self, optimizer, step_size, gamma):
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

