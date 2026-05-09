"""Function-space prior builder."""

from src.builders.base import BaseBuilder
from src.registry import PRIOR_FS_REGISTRY
import src.prior_fs  # noqa: F401


class PriorFSBuilder(BaseBuilder):
    """Builds function-space prior distributions."""

    def build(self):
        cls = PRIOR_FS_REGISTRY.get(self.config.name)
        params = self.config.get('params', {}) or {}
        return cls(**params)
