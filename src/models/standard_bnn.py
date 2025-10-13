"""
Standard Bayesian Neural Network with MCMC sampling using posteriors.
"""

import torch
import torch.nn as nn
from torch import func
import posteriors
from typing import Dict, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm

from .base_bnn import BaseBNN


class StandardBNN(BaseBNN):
    """
    Standard Bayesian Neural Network using MCMC sampling with posteriors.
    
    This implementation uses stochastic gradient MCMC methods from the posteriors
    library to approximate the posterior distribution over network parameters.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        num_classes: int,
        prior_std: float = 1.0,
        temperature: float = 1.0,
        mcmc_method: str = "sgld",
        device: str = 'auto'
    ):
        """
        Initialize the Standard BNN.
        
        Args:
            model: PyTorch model architecture
            num_classes: Number of output classes
            prior_std: Standard deviation of the prior distribution
            temperature: Temperature for tempering the posterior
            mcmc_method: MCMC method to use ('sgld', 'sghmc', 'sgnht', 'baoa')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        super().__init__(model, num_classes, device)
        self.prior_std = prior_std
        self.temperature = temperature
        self.mcmc_method = mcmc_method
        self.transform = None
        
        # Determine if this model needs flattened input (MLP) or structured input (CNN)
        self._needs_flattened_input = self._check_if_needs_flattening(model)
    
    def _check_if_needs_flattening(self, model: nn.Module) -> bool:
        """
        Check if the model needs flattened input (MLP) or structured input (CNN).
        
        Args:
            model: PyTorch model to inspect
            
        Returns:
            True if model needs flattened input, False if it needs structured input
        """
        # Check if the first layer is a Linear layer (MLP) or Conv layer (CNN)
        if isinstance(model, nn.Sequential):
            first_layer = model[0]
        else:
            # For non-sequential models, check the first child module
            first_layer = next(model.children(), None)
        
        if isinstance(first_layer, nn.Linear):
            return True  # MLP needs flattened input
        elif isinstance(first_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return False  # CNN needs structured input
        else:
            # Default to flattening for unknown architectures
            return True
        
    def build_transform(
        self, 
        lr: float = 1e-3, 
        num_data: int = 50000,
        **kwargs
    ):
        """
        Build the posteriors transform for MCMC sampling.
        
        Args:
            lr: Learning rate for MCMC
            num_data: Total number of training data points (for proper scaling)
            **kwargs: Additional arguments for specific MCMC methods
                - beta: Gradient noise coefficient for SGLD/SGHMC/SGNHT/BAOA (default: 0.0)
                - alpha: Friction coefficient for SGHMC/SGNHT/BAOA (default: 0.01)
                - sigma: Standard deviation of momenta target distribution for SGHMC/SGNHT/BAOA (default: 1.0)
                - momenta: Initial momenta for SGHMC/SGNHT/BAOA (default: None)
                - xi: Initial thermostat value for SGNHT (default: alpha)
        """
        # log-posterior is averaged per sample.
        # To target a dataset-level tempered posterior p(θ|D)^(1/T),
        # the sampler must see Teff = T / N.
        self.num_data = num_data
        effective_temperature = self.temperature / num_data
        
        # Extract method-specific parameters from kwargs
        beta = kwargs.get('beta', 0.0)  # Gradient noise coefficient
        alpha = kwargs.get('alpha', 0.01)  # Friction coefficient
        sigma = kwargs.get('sigma', 1.0)  # Standard deviation of momenta target distribution
        momenta = kwargs.get('momenta', None)  # Initial momenta
        xi = kwargs.get('xi', alpha)  # Initial thermostat value for SGNHT
        
        # Build transform based on MCMC method
        if self.mcmc_method == "sgld":
            # Stochastic Gradient Langevin Dynamics
            # Reference: https://normal-computing.github.io/posteriors/api/sgmcmc/sgld/
            self.transform = posteriors.sgmcmc.sgld.build(
                log_posterior=self.log_posterior,
                lr=lr,
                beta=beta,
                temperature=effective_temperature
            )
        elif self.mcmc_method == "sghmc":
            # Stochastic Gradient Hamiltonian Monte Carlo
            # Reference: https://normal-computing.github.io/posteriors/api/sgmcmc/sghmc/
            try:
                self.transform = posteriors.sgmcmc.sghmc.build(
                    log_posterior=self.log_posterior,
                    lr=lr,
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    temperature=effective_temperature,
                    momenta=momenta
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Try without momenta parameter
                    print(f"Warning: SGHMC parameter issue, trying without momenta: {e}")
                    self.transform = posteriors.sgmcmc.sghmc.build(
                        log_posterior=self.log_posterior,
                        lr=lr,
                        alpha=alpha,
                        beta=beta,
                        sigma=sigma,
                        temperature=effective_temperature
                    )
                else:
                    raise e
        elif self.mcmc_method == "sgnht":
            # Stochastic Gradient Nosé-Hoover Thermostat
            # Reference: https://normal-computing.github.io/posteriors/api/sgmcmc/sgnht/
            try:
                self.transform = posteriors.sgmcmc.sgnht.build(
                    log_posterior=self.log_posterior,
                    lr=lr,
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    temperature=effective_temperature,
                    momenta=momenta,
                    xi=xi
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Try with minimal parameters
                    print(f"Warning: SGNHT parameter issue, trying minimal set: {e}")
                    self.transform = posteriors.sgmcmc.sgnht.build(
                        log_posterior=self.log_posterior,
                        lr=lr,
                        temperature=effective_temperature
                    )
                else:
                    raise e
        elif self.mcmc_method == "baoa":
            # Bayesian Averaging Over Architectures (BAOA integrator for SGHMC)
            # Reference: https://normal-computing.github.io/posteriors/api/sgmcmc/baoa/
            # BAOA may have different parameters than other methods
            try:
                # Try with full parameter set first
                self.transform = posteriors.sgmcmc.baoa.build(
                    log_posterior=self.log_posterior,
                    lr=lr,
                    alpha=alpha,
                    sigma=sigma,
                    temperature=effective_temperature,
                    momenta=momenta
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Try with minimal parameter set
                    print(f"Warning: BAOA doesn't accept all parameters, trying minimal set: {e}")
                    self.transform = posteriors.sgmcmc.baoa.build(
                        log_posterior=self.log_posterior,
                        lr=lr,
                        temperature=effective_temperature
                    )
                else:
                    raise e
        else:
            # Default to SGLD for unknown methods
            print(f"Warning: Unknown MCMC method '{self.mcmc_method}', falling back to SGLD")
            self.transform = posteriors.sgmcmc.sgld.build(
                self.log_posterior,
                lr=lr,
                beta=beta,
                temperature=effective_temperature
            )
    
    def log_posterior(
        self, 
        params: Dict[str, torch.Tensor], 
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log posterior probability.
        
        Args:
            params: Model parameters
            batch: (images, labels) batch
            
        Returns:
            Tuple of (log_posterior_value, model_output)
        """
        images, labels = batch
        
        # Flatten images only if the model needs flattened input (MLP)
        if self._needs_flattened_input and len(images.shape) > 2:
            images = images.view(images.size(0), -1)
        
        # Forward pass
        output = func.functional_call(self.model, params, images)
        
        # Log posterior calculation (following posteriors library example)
        log_post_val = (
            -nn.functional.cross_entropy(output, labels)
            + posteriors.diag_normal_log_prob(params) / getattr(self, "num_data", len(images))
        )
        
        return log_post_val, output
    
    def _move_state_to_device(self, state):
        """
        Move posteriors state to the correct device.
        
        Args:
            state: Posteriors state object
            
        Returns:
            State moved to device
        """
        if hasattr(state, 'params') and isinstance(state.params, dict):
            # Move parameters to device
            moved_params = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in state.params.items()}
            
            # Handle different state types based on their attributes
            state_kwargs = {
                'params': moved_params,
                'log_posterior': state.log_posterior.to(self.device) if torch.is_tensor(state.log_posterior) else state.log_posterior,
                'step': state.step.to(self.device) if torch.is_tensor(state.step) else state.step
            }
            
            # Add momenta if present (SGHMC, SGNHT, BAOA)
            if hasattr(state, 'momenta') and state.momenta is not None:
                moved_momenta = {k: v.to(self.device) if torch.is_tensor(v) else v 
                               for k, v in state.momenta.items()}
                state_kwargs['momenta'] = moved_momenta
            
            # Add xi if present (SGNHT)
            if hasattr(state, 'xi') and state.xi is not None:
                # Handle xi - it could be a dict or a scalar tensor
                if isinstance(state.xi, dict):
                    moved_xi = {k: v.to(self.device) if torch.is_tensor(v) else v 
                              for k, v in state.xi.items()}
                else:
                    moved_xi = state.xi.to(self.device) if torch.is_tensor(state.xi) else state.xi
                state_kwargs['xi'] = moved_xi
            
            # Create new state with the appropriate arguments
            new_state = type(state)(**state_kwargs)
            return new_state
        elif isinstance(state, dict):
            # State is a dictionary directly
            return {k: v.to(self.device) if torch.is_tensor(v) else v 
                   for k, v in state.items()}
        else:
            # Try to move the entire state if it's a tensor
            try:
                return state.to(self.device) if hasattr(state, 'to') else state
            except:
                return state
    
    def fit(
        self, 
        train_loader, 
        num_epochs: int = 100,
        lr: float = 1e-3,
        num_burn_in: int = 50,
        verbose: bool = True,
        **kwargs
    ):
        """
        Train the BNN using MCMC sampling.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            num_burn_in: Number of burn-in epochs (samples to discard)
            verbose: Whether to print training progress
        """
        # Get dataset size for proper temperature scaling
        num_data = len(train_loader.dataset)
        
        # Build transform if not already built
        if self.transform is None:
            self.build_transform(lr=lr, num_data=num_data, **kwargs)
        
        # Initialize state - ensure parameters are on correct device
        device_params = {name: param.to(self.device) for name, param in self.params.items()}
        self.state = self.transform.init(device_params)
        
        # Ensure state is on correct device
        self.state = self._move_state_to_device(self.state)
        
        # Track samples after burn-in
        self.posterior_samples = []
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training BNN", disable=not verbose)
        
        for epoch in epoch_pbar:
            num_batches = 0
            
            # Create batch progress bar
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False, disable=not verbose)
            
            for batch in batch_pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Update state
                self.state, info = self.transform.update(self.state, batch)
                
                # Ensure state remains on correct device after update
                self.state = self._move_state_to_device(self.state)
                
                num_batches += 1
                
                # Collect samples after burn-in
                if epoch >= num_burn_in:
                    # Sample current parameters
                    sample = self.sample_posterior()
                    self.posterior_samples.append(sample)
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'Batch': f"{num_batches}/{len(train_loader)}",
                    'Samples': len(self.posterior_samples) if epoch >= num_burn_in else 0
                })
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Samples Collected': len(self.posterior_samples),
                'Burn-in': 'Complete' if epoch >= num_burn_in else f"{epoch}/{num_burn_in}"
            })
        
        if verbose:
            print(f"Training completed! Collected {len(self.posterior_samples)} posterior samples.")
    
    def sample_posterior(self) -> Dict[str, torch.Tensor]:
        """
        Sample parameters from the current posterior state.
        
        Returns:
            Dictionary of sampled parameters
        """
        if self.state is None:
            raise ValueError("Model must be trained first!")
        
        # For MCMC methods, the current state contains the current sample
        if hasattr(self.state, 'params'):
            return self.state.params
        else:
            # Fallback: return the state itself if it's already parameters
            return self.state
    
    def predict_batch(
        self, 
        batch: torch.Tensor, 
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for a batch with uncertainty quantification.
        
        Args:
            batch: Input batch
            num_samples: Number of posterior samples (if None, use all collected samples)
            
        Returns:
            Tuple of (mean_predictions, epistemic_uncertainty)
        """
        if not hasattr(self, 'posterior_samples') or len(self.posterior_samples) == 0:
            raise ValueError("No posterior samples available! Train the model first.")
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Use all collected samples or a random subset if num_samples is set.
        samples_to_use = self.posterior_samples
        if num_samples is not None and num_samples < len(self.posterior_samples):
            indices = np.random.choice(len(self.posterior_samples), num_samples, replace=False)
            samples_to_use = [self.posterior_samples[i] for i in indices]
        
        predictions = []
        
        for sample_params in samples_to_use:
            with torch.no_grad():
                output = func.functional_call(self.model, sample_params, batch)
                predictions.append(torch.softmax(output, dim=1))
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Predictive mean over posterior samples.
        mean_pred = torch.mean(predictions, dim=0)
        
        # Simple epistemic uncertainty. Variance of class probabilities across samples, summed over classes.
        # this is a variance based uncertainty, different from entropy based MI (total - aleatoric).
        epistemic_uncertainty = torch.var(predictions, dim=0).sum(dim=1)
        
        return mean_pred, epistemic_uncertainty
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        Evaluate the BNN on test data using all collected posterior samples.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self, 'posterior_samples') or len(self.posterior_samples) == 0:
            raise ValueError("No posterior samples available! Train the model first.")
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        # Sample-weighted accumulators. Results are dataset-level averages.
        sum_loss = 0.0
        sum_total_u = 0.0
        sum_aleatoric_u = 0.0
        sum_epistemic_u = 0.0
        total_N = 0
        
        num_samples = len(self.posterior_samples)
        
        # Progress bar over test batches.
        eval_pbar = tqdm(test_loader, desc="Evaluating BNN", leave=False)
        
        eps = 1e-8
        batch_idx = 0
        
        for batch in eval_pbar:
            images, labels = batch
            labels = labels.to(self.device)
            images = images.to(self.device)
            B = labels.size(0)
            
            # Predictive probabilities per posterior sample. Shape [num_samples, batch_size, num_classes]
            probs_list = []
            for sample_params in self.posterior_samples:
                with torch.no_grad():
                    logits = func.functional_call(self.model, sample_params, images)
                    probs_list.append(torch.softmax(logits, dim=1))
            probs = torch.stack(probs_list, dim=0)
            
            # Posterior predictive mean. E_s[p(y|x, θ_s)] with small clamp for stability.
            expected_probs = probs.mean(dim=0)
            expected_probs = torch.clamp(expected_probs, min=eps)
            expected_probs = expected_probs / expected_probs.sum(dim=1, keepdim=True)
            
            # Mean prediction and uncertainty
            all_predictions.append(expected_probs)
            all_labels.append(labels)
            epistemic_uncertainty = torch.var(probs, dim=0).sum(dim=1)
            all_uncertainties.append(epistemic_uncertainty)
            
            # Loss. negative log-likelihood of expected probs
            loss_batch = torch.nn.functional.nll_loss(torch.log(expected_probs), labels, reduction='mean')
            
            # Entropy-based decomposition:
            #   total = H(E_s[p])
            #   aleatoric = E_s[H(p_s)]
            #   epistemic = total - aleatoric
            total_u = -(expected_probs * torch.log(expected_probs + eps)).sum(dim=1).mean()
            ent_per_sample = -(probs * torch.log(probs + eps)).sum(dim=2)  # [S, B]
            aleatoric_u = ent_per_sample.mean(dim=0).mean()
            epistemic_u = (total_u - aleatoric_u)
            
            # Sample weighted accumulation.
            sum_loss += float(loss_batch.item()) * B
            sum_total_u += float(total_u.item()) * B
            sum_aleatoric_u += float(aleatoric_u.item()) * B
            sum_epistemic_u += float(epistemic_u.item()) * B
            total_N += B
            
            batch_idx += 1
            eval_pbar.set_postfix({
                'Batch': f"{batch_idx}/{len(test_loader)}",
                'Samples': f"{total_N}/{len(test_loader.dataset)}",
                'Posterior': num_samples
            })
        
        # Concatenate and compute aggregate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        
        accuracy = self.compute_accuracy(all_predictions, all_labels)
        ece = self.compute_ece(all_predictions, all_labels)
        avg_uncertainty = all_uncertainties.mean().item()
        
        # Final sample-weighted metrics
        loss = sum_loss / max(1, total_N)
        total_uncertainty = sum_total_u / max(1, total_N)
        aleatoric_uncertainty = sum_aleatoric_u / max(1, total_N)
        epistemic_uncertainty = sum_epistemic_u / max(1, total_N)
        
        return {
            'accuracy': accuracy,
            'ece': ece,
            'avg_uncertainty': avg_uncertainty,
            'num_samples': num_samples,
            'test_samples': len(all_predictions),
            'loss': loss,
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
        }
