"""SGLD optimizer using posteriors library."""

from posteriors.sgmcmc import sgld
from src.registry import OPTIMIZER_REGISTRY
from .base import BNNOptimizer


@OPTIMIZER_REGISTRY.register("sgld")
class SGLDOptimizer(BNNOptimizer):
    """Stochastic Gradient Langevin Dynamics optimizer."""
    
    def __init__(self, lr, temperature=1.0, beta=0.0):
        self.lr = lr
        self.temperature = temperature
        self.beta = beta
    
    def _build_transform(self):
        """Build SGLD transform from posteriors library."""
        return sgld.build(
            log_posterior=self.log_posterior,
            lr=self.lr,
            beta=self.beta,
            temperature=self.scaled_temperature
        )
    
    def state_dict(self):
        """Save optimizer state for checkpointing."""
        return {
            'params': self.params,
            'posteriors_state': {
                'params': {k: v.cpu().clone() for k, v in self.state.params.items()},
                'log_posterior': self.state.log_posterior.cpu().clone() if hasattr(self.state.log_posterior, 'cpu') else self.state.log_posterior,
                'step': self.state.step.cpu().clone() if hasattr(self.state.step, 'cpu') else self.state.step
            },
            'lr': self.lr,
            'temperature': self.temperature,
            'beta': self.beta
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        from collections import OrderedDict
        from posteriors.sgmcmc.sgld import SGLDState
        
        # Load parameter values into model
        for name, param_value in state_dict['posteriors_state']['params'].items():
            self.model.state_dict()[name].copy_(param_value)
        
        # Recreate self.params as references to model parameters 
        self.params = OrderedDict(self.model.named_parameters())
        
        # Restore posteriors state with the actual model parameter references
        self.state = SGLDState(
            params=self.params,
            log_posterior=state_dict['posteriors_state']['log_posterior'],
            step=state_dict['posteriors_state']['step']
        )
