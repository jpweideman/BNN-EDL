"""Sampler builder."""

from src.builders.base import BaseBuilder
from src.registry import SAMPLER_REGISTRY
import src.samplers  # noqa: F401


class SamplerBuilder(BaseBuilder):
    """Builds MCMC samplers from configuration."""

    def build(self, model_parameters, model=None, likelihood_fn=None, prior_fn=None, num_data=None, prior_fs_fn=None):
        sampler_cls = SAMPLER_REGISTRY.get(self.config.name)
        params = self.config.get('params', {}) or {}
        return sampler_cls(**params)(
            model_parameters,
            model=model,
            likelihood_fn=likelihood_fn,
            prior_fn=prior_fn,
            num_data=num_data,
            prior_fs_fn=prior_fs_fn,
        )
