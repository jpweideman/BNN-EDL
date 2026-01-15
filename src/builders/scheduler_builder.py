from src.builders.base import BaseBuilder
from src.registry import SCHEDULER_REGISTRY
import src.schedulers  # noqa: F401 # Triggers registration of schedulers


class SchedulerBuilder(BaseBuilder):
    def build(self, optimizer):
        if self.config is None:
            return None
        
        scheduler_cls = SCHEDULER_REGISTRY.get(self.config.name)
        params = {k: v for k, v in self.config.items() if k not in ['name', 'enabled']}
        return scheduler_cls(optimizer, **params).scheduler

