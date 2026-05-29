"""SGLRW optimizer using posteriors library."""

from posteriors.sgmcmc import sglrw
from src.registry import SAMPLER_REGISTRY
from .base import BNNOptimizer
from .log_posteriors.log_posterior import LogPosterior


@SAMPLER_REGISTRY.register("sglrw")
class SGLRWOptimizer(BNNOptimizer):
    """Stochastic Gradient Lattice Random Walk optimizer."""

    def __init__(self, lr, temperature=1.0):
        self.lr = lr
        self.temperature = temperature

    def _build_log_posterior(self, model, likelihood_fn, prior_fn, prior_fs_fn=None):
        return LogPosterior(model, likelihood_fn, prior_fn, prior_fs_fn)

    def _build_transform(self):
        return sglrw.build(
            log_posterior=self.log_posterior,
            lr=self.lr,
            temperature=self.scaled_temperature,
        )

    def state_dict(self):
        return {
            'params': self.params,
            'posteriors_state': {
                'params': {k: v.cpu().clone() for k, v in self.state.params.items()},
                'log_posterior': self.state.log_posterior.cpu().clone() if hasattr(self.state.log_posterior, 'cpu') else self.state.log_posterior,
                'step': self.state.step.cpu().clone() if hasattr(self.state.step, 'cpu') else self.state.step,
            },
            'lr': self.lr,
            'temperature': self.temperature,
        }

    def load_state_dict(self, state_dict):
        from collections import OrderedDict
        from posteriors.sgmcmc.sglrw import SGLRWState
        for name, param_value in state_dict['posteriors_state']['params'].items():
            self.model.state_dict()[name].copy_(param_value)
        self.params = OrderedDict(self.model.named_parameters())
        self.state = SGLRWState(
            params=self.params,
            log_posterior=state_dict['posteriors_state']['log_posterior'],
            step=state_dict['posteriors_state']['step'],
        )
