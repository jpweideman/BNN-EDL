"""Optimizer builder."""

from src.builders.base import BaseBuilder
from src.registry import OPTIMIZER_REGISTRY
import src.optimizers  # noqa: F401


class OptimizerBuilder(BaseBuilder):
    """Builds standard optimizers from configuration."""

    def build(self, model_parameters, model=None, loss_fn=None, likelihood_fn=None, prior_fn=None, **kwargs):
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.config.name)
        params = self.config.get('params', {}) or {}
        return optimizer_cls(
            model_parameters,
            model=model,
            loss_fn=loss_fn,
            likelihood_fn=likelihood_fn,
            prior_fn=prior_fn,
            **params
        )
