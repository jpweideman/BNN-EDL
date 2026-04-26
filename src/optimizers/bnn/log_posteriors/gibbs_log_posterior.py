"""Gibbs log posterior for block-coordinate inference in the EDL-BNN model."""

import torch
import torch.func as func
from torch.distributions import Gamma


def gibbs_sample_f(alpha: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Sample the exact Gibbs conditional p(f | w, y) via Dir-Cat conjugacy.

    Adds 1 to the true class concentration and samples from the resulting
    Dirichlet. Must be called inside torch.no_grad() with detached alpha.

    Args:
        alpha:   concentration parameters [B, C], alpha_c >= 1, detached
        y_true:  class labels [B], dtype long

    Returns:
        f_samples: shape [B, C], each row sums to 1
    """
    device = alpha.device
    alpha_posterior = alpha.cpu().clone()
    alpha_posterior[torch.arange(len(y_true)), y_true.cpu()] += 1.0
    g = Gamma(alpha_posterior, torch.ones_like(alpha_posterior)).sample()
    f = g / g.sum(dim=-1, keepdim=True)
    return f.to(device)


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
