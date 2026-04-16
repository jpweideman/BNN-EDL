"""Gibbs log posterior for block-coordinate inference in the EDL-BNN model."""

import torch
import torch.func as func
from src.likelihoods.dirichlet_joint import gibbs_sample_f


class GibbsLogPosterior:
    """Log posterior for block-coordinate SGLD with Gibbs sampling of the latent simplex.

    Draws one sample from the Gibbs conditional p(f | w, y) per step
    and evaluates the joint log-likelihood at that sample.
    """

    def __init__(self, model, likelihood_fn, prior_fn):
        self.model = model
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn
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

        alpha = func.functional_call(self.model, params, x)
        with torch.no_grad():
            f_samples = gibbs_sample_f(alpha.detach(), y)
        log_likelihood = self.likelihood_fn(alpha, f_samples)
        log_prior = self.prior_fn(params)
        log_posterior = log_likelihood + log_prior

        # Store for logging
        self.last_log_likelihood = log_likelihood.item()
        self.last_log_prior = log_prior.item()
        self.last_log_posterior = log_posterior.item()

        return log_posterior, torch.tensor([])
