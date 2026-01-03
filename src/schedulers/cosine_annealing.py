import torch.optim.lr_scheduler as lr_scheduler
from src.registry import SCHEDULER_REGISTRY


@SCHEDULER_REGISTRY.register("cosine_annealing")
class CosineAnnealingLRScheduler:
    def __init__(self, optimizer, T_max, eta_min):
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

