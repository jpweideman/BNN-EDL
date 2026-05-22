"""Function-space prior builder."""

from src.builders.base import BaseBuilder
from src.registry import PRIORS_FS_REGISTRY
import src.priors_fs  # noqa: F401


class PriorsFSBuilder(BaseBuilder):
    """Builds function-space prior distributions."""

    def build(self):
        cls = PRIORS_FS_REGISTRY.get(self.config.name)
        params = self.config.get('params', {}) or {}
        return cls(**params)
