"""SGHMC optimizer using posteriors library."""

from posteriors.sgmcmc import sghmc
from src.registry import OPTIMIZER_REGISTRY
from .base import BNNOptimizer


@OPTIMIZER_REGISTRY.register("sghmc")
class SGHMCOptimizer(BNNOptimizer):
    """Stochastic Gradient Hamiltonian Monte Carlo optimizer."""
    
    def __init__(self, lr, temperature=1.0, alpha=0.01, beta=0.0, sigma=1.0, momenta=0.0):
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.momenta = momenta
    
    def _build_transform(self):
        """Build SGHMC transform from posteriors library."""
        return sghmc.build(
            log_posterior=self.log_posterior,
            lr=self.lr,
            alpha=self.alpha,
            beta=self.beta,
            sigma=self.sigma,
            temperature=self.scaled_temperature,
            momenta=self.momenta
        )
    
    def state_dict(self):
        """Save optimizer state for checkpointing."""
        return {
            'params': self.params,
            'posteriors_state': {
                'params': {k: v.cpu().clone() for k, v in self.state.params.items()},
                'momenta': {k: v.cpu().clone() for k, v in self.state.momenta.items()},
                'log_posterior': self.state.log_posterior.cpu().clone() if hasattr(self.state.log_posterior, 'cpu') else self.state.log_posterior,
                'step': self.state.step.cpu().clone() if hasattr(self.state.step, 'cpu') else self.state.step
            },
            'lr': self.lr,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma': self.sigma,
            'momenta': self.momenta
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        from collections import OrderedDict
        from posteriors.sgmcmc.sghmc import SGHMCState
        
        # Load parameter values into model
        for name, param_value in state_dict['posteriors_state']['params'].items():
            self.model.state_dict()[name].copy_(param_value)
        
        # Recreate self.params as references to model parameters 
        self.params = OrderedDict(self.model.named_parameters())
        
        # Restore posteriors state with the actual model parameter references
        self.state = SGHMCState(
            params=self.params,
            momenta=state_dict['posteriors_state']['momenta'],
            log_posterior=state_dict['posteriors_state']['log_posterior'],
            step=state_dict['posteriors_state']['step']
        )
