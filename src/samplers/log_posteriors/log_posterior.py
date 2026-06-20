"""Log posterior construction for Bayesian inference."""

import torch
import torch.func as func


class LogPosterior:
    """
    Computes log p(θ|D) = log p(D|θ) + log p(θ) [+ log p_fs(f(x;θ))] for BNN sampling.

    Args:
        model:        PyTorch model
        likelihood_fn: Returns log p(D|θ) given predictions and targets
        prior_fn:     Returns log p(θ) given parameters (must be pre-scaled by 1/N)
        prior_fs_fn: Optional function-space prior; takes model output alpha and returns scalar
    """

    def __init__(self, model, likelihood_fn, prior_fn, prior_fs_fn=None):
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn
        self.prior_fs_fn = prior_fs_fn
        self.last_log_likelihood = None
        self.last_log_prior = None
        self.last_log_prior_fs = None
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

        model_output = func.functional_call(self.model, params, x)
        log_likelihood = self.likelihood_fn(model_output, y)
        log_prior = self.prior_fn(params)
        log_prior_fs = self.prior_fs_fn(model_output) if self.prior_fs_fn is not None else 0.0
        log_posterior = log_likelihood + log_prior + log_prior_fs

        # Store for logging
        self.last_log_likelihood = log_likelihood.item()
        self.last_log_prior = log_prior.item()
        self.last_log_prior_fs = log_prior_fs.item() if self.prior_fs_fn is not None else 0.0
        self.last_log_posterior = log_posterior.item()

        return log_posterior, torch.tensor([])
