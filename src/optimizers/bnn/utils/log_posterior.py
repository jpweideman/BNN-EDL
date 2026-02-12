"""Log posterior construction for Bayesian inference."""

import torch
import torch.func as func


class LogPosterior:
    """
    Computes log p(θ|D) = log p(D|θ) + log p(θ) for BNN sampling.
    
    Args:
        model: PyTorch model
        likelihood_fn: Returns log p(D|θ) given predictions and targets
        prior_fn: Returns log p(θ) given parameters (must be pre-scaled by 1/N)
    """
    
    def __init__(self, model, likelihood_fn, prior_fn):
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn
        
        # Store for logging
        self.last_log_likelihood = None
        self.last_log_prior = None
        self.last_log_posterior = None
        
    def __call__(self, params, batch):
        """
        Compute log posterior for a batch.
        
        Args:
            params: Model parameters dict
            batch: (x, y) tuple
            
        Returns:
            Tuple[log_posterior, aux]: (scalar tensor, empty tensor)
        """
        x, y = batch
        device = next(iter(params.values())).device
        x, y = x.to(device), y.to(device)
        
        y_pred = func.functional_call(self.model, params, x)
        log_likelihood = self.likelihood_fn(y_pred, y)
        
        # Pass batch and model to prior if it needs them
        if hasattr(self.prior_fn, 'model'):
            self.prior_fn.model = self.model
            self.prior_fn._batch = batch
        
        log_prior = self.prior_fn(params)
        log_posterior = log_likelihood + log_prior
        
        # Store for logging
        self.last_log_likelihood = log_likelihood.item()
        self.last_log_prior = log_prior.item()
        self.last_log_posterior = log_posterior.item()
        
        return log_posterior, torch.tensor([])

