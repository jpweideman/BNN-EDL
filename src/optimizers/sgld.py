"""SGLD optimizer using posteriors library."""

from posteriors.sgmcmc import sgld
from collections import OrderedDict
from src.registry import OPTIMIZER_REGISTRY
from src.optimizers.bnn.log_posterior import LogPosterior

@OPTIMIZER_REGISTRY.register("sgld")
class SGLDOptimizer:
    """Stochastic Gradient Langevin Dynamics optimizer."""
    
    def __init__(self, lr, temperature=1.0, beta=0.0):
        self.lr = lr
        self.temperature = temperature
        self.beta = beta
        
    def __call__(self, model_parameters, model=None, likelihood_fn=None, prior_fn=None, **kwargs):
        self.model = model
        self.params = OrderedDict(model.named_parameters())
        
        # Create log posterior with likelihood and prior functions
        self.log_posterior = LogPosterior(model, likelihood_fn, prior_fn)
        
        # Build SGLD transform
        self.transform = sgld.build(
            log_posterior=self.log_posterior,
            lr=self.lr,
            beta=self.beta,
            temperature=self.temperature
        )
        self.state = self.transform.init(self.params)
        
        return self
    
    def step(self, batch):
        """Perform one SGLD sampling step."""
        self.state, aux = self.transform.update(self.state, batch)
        self.params = self.state.params
        for name, param in self.params.items():
            self.model.state_dict()[name].copy_(param)
    
    def get_last_metrics(self):
        """Get BNN metrics from last step for logging."""
        return {
            'log_likelihood': self.log_posterior.last_log_likelihood,
            'log_prior': self.log_posterior.last_log_prior,
            'log_posterior': self.log_posterior.last_log_posterior,
        }
    
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
        from posteriors.sgmcmc.sgld import SGLDState
        
        self.params = state_dict['params']
        
        # Restore posteriors state
        self.state = SGLDState(
            params=state_dict['posteriors_state']['params'],
            log_posterior=state_dict['posteriors_state']['log_posterior'],
            step=state_dict['posteriors_state']['step']
        )
        
        # Update model parameters
        for name, param in self.params.items():
            self.model.state_dict()[name].copy_(param)

