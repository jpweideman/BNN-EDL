"""Generalized Bayes likelihood wrapping the evidential classification loss.

Under Generalized Bayes (Bissiri et al. 2016), any loss L defines a valid
Gibbs posterior: p(θ|D) ∝ exp(-η * L(θ,D)) * p(θ).
Negating the EDL loss gives the pseudo-log-likelihood for SGLD to maximize.
The SGLD temperature parameter plays the role of η.
"""

from edl_pytorch import evidential_classification as edl_loss
from src.registry import LIKELIHOOD_REGISTRY


@LIKELIHOOD_REGISTRY.register("evidential_generalized")
class EvidentialGeneralizedLikelihood:
    """Generalized Bayes pseudo-likelihood from the evidential classification loss.

    Returns -edl_loss(alpha, y) so SGLD (which maximizes) samples from the
    Gibbs posterior induced by dirichlet_mse + lamb * kl_reg.
    """

    def __init__(self, lamb):
        self.lamb = lamb

    def __call__(self, alpha, y_true):
        """
        Args:
            alpha: Dirichlet parameters, shape [B, C]
            y_true: Ground truth labels, shape [B]

        Returns:
            Negated EDL loss scalar to be maximized by SGLD.
        """
        return -edl_loss(alpha, y_true, lamb=self.lamb)
