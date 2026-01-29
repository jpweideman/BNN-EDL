"""Abstract base class for BNN optimizers."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from .utils.log_posterior import LogPosterior


class BNNOptimizer(ABC):
    """Abstract base class for BNN optimizers.
    
    All BNN optimizers must inherit from this class to ensure they have
    the required interface and can be detected via isinstance checks.
    """
    
    def __call__(self, model_parameters, model=None, likelihood_fn=None, prior_fn=None, num_data=None, **kwargs):
        """Initialize the BNN optimizer with model and inference functions.
        
        Args:
            model_parameters: Model parameters to optimize (not used directly, kept for compatibility)
            model: PyTorch model
            likelihood_fn: Likelihood function that computes log p(D|θ)
            prior_fn: Prior function that computes log p(θ)
            num_data: Number of training data points (for temperature scaling)
            **kwargs: Additional arguments 
            
        Returns:
            self: The initialized optimizer instance
        """
        self.model = model
        self.params = OrderedDict(model.named_parameters())
        self.num_data = num_data
        
        # Scale temperature by 1/num_data 
        if hasattr(self, 'temperature'):
            self.scaled_temperature = self.temperature / self.num_data
        
        # Create log posterior combining likelihood and prior
        self.log_posterior = LogPosterior(model, likelihood_fn, prior_fn)
        
        # Build the transform. Implemented by subclass
        self.transform = self._build_transform()
        
        # Initialize the optimizer state
        self.state = self.transform.init(self.params)
        
        return self
    
    @abstractmethod
    def _build_transform(self):
        """Build the posteriors transform for this optimizer.
        
        This method must be implemented by subclasses to return the
        appropriate transform from the posteriors library (e.g., sgld.build()).
        
        Returns:
            Transform object from posteriors library
        """
        pass
    
    def step(self, batch):
        """Perform one optimization step.
        
        This is common logic for all BNN optimizers using the posteriors library.
        
        Args:
            batch: Training batch (x, y)
        """
        self.state, aux = self.transform.update(self.state, batch)
        self.params = self.state.params
        for name, param in self.params.items():
            self.model.state_dict()[name].copy_(param)
    
    def get_last_metrics(self):
        """Get metrics from the last optimization step.
        
        Returns:
            dict: Dictionary with log_likelihood, log_prior, log_posterior
        """
        return {
            'log_likelihood': self.log_posterior.last_log_likelihood,
            'log_prior': self.log_posterior.last_log_prior,
            'log_posterior': self.log_posterior.last_log_posterior,
        }
    
    @abstractmethod
    def state_dict(self):
        """Save optimizer state for checkpointing.
        
        Returns:
            dict: State dictionary
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint.
        
        Args:
            state_dict: State dictionary to load
        """
        pass
