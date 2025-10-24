"""
Evidential Deep Learning Bayesian Neural Network with MCMC sampling.
Combines EDL output layers with BNN posterior sampling.
"""

import torch
import torch.nn as nn
from torch import func
import posteriors
from typing import Dict, Tuple, Any, Optional
import numpy as np
import os
from tqdm import tqdm

from .base_bnn import BaseBNN
from utils.training import CosineLR

from edl_pytorch import evidential_classification

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class EDLBNN(BaseBNN):
    """
    Evidential Deep Learning Bayesian Neural Network.
    
    This implementation combines:
    - EDL (Evidential Deep Learning) for uncertainty quantification via Dirichlet distributions
    - BNN posterior sampling via MCMC methods (SGLD, SGHMC, etc.)
    
    The model uses a Dirichlet output layer to produce evidential parameters (alpha),
    which are then used for both training (via evidential loss) and inference (via posterior sampling).
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        num_classes: int,
        prior_std: float = 1.0,
        temperature: float = 1.0,
        mcmc_method: str = "sgld",
        edl_lambda: float = 0.001,
        device: str = 'auto'
    ):
        """
        Initialize the EDL BNN.
        
        Args:
            model: PyTorch model architecture (should end with Dirichlet layer)
            num_classes: Number of output classes
            prior_std: Standard deviation of the prior distribution
            temperature: Temperature for tempering the posterior
            mcmc_method: MCMC method to use ('sgld', 'sghmc', 'sgnht', 'baoa')
            edl_lambda: Regularization coefficient for EDL loss (default: 0.001)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        super().__init__(model, num_classes, device)
        self.prior_std = prior_std
        self.temperature = temperature
        self.mcmc_method = mcmc_method
        self.edl_lambda = edl_lambda
        self.transform = None
        
        # Determine if this model needs flattened input (MLP) or structured input (CNN)
        self._needs_flattened_input = self._check_if_needs_flattening(model)
    
    def get_annealed_edl_lambda(self, epoch: int) -> float:
        """
        Compute the annealed EDL lambda for KL regularization.
        
        Anneals from 0 to self.edl_lambda over edl_kl_anneal_epochs.
        After annealing period, stays constant at self.edl_lambda.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Annealed lambda value for this epoch
        """
        if not hasattr(self, 'edl_kl_anneal_epochs'):
            # If annealing not configured, return constant lambda
            return self.edl_lambda
        
        # Linear annealing: 0 → edl_lambda over first edl_kl_anneal_epochs
        if epoch < self.edl_kl_anneal_epochs:
            return self.edl_lambda * (epoch / self.edl_kl_anneal_epochs)
        else:
            return self.edl_lambda
    
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
        Compute log posterior probability using EDL loss.
        
        Args:
            params: Model parameters
            batch: (images, labels) batch
            
        Returns:
            Tuple of (log_posterior_value, alpha_output)
        """
        images, labels = batch
        
        # Flatten images only if the model needs flattened input (MLP)
        if self._needs_flattened_input and len(images.shape) > 2:
            images = images.view(images.size(0), -1)
        
        # Forward pass - outputs Dirichlet alpha parameters
        alpha = func.functional_call(self.model, params, images)
        
        # EDL loss (negative because we want log posterior, not loss)
        # The EDL loss combines classification loss + KL regularization
        # Use annealed lambda if available, otherwise use constant lambda
        current_lambda = self.get_annealed_edl_lambda(getattr(self, '_current_epoch', 0))
        edl_loss = evidential_classification(alpha, labels, lamb=current_lambda)
        
        # Log posterior calculation:
        # - Likelihood: negative EDL loss
        # - Prior: Gaussian prior over parameters
        log_post_val = (
            -edl_loss
            + posteriors.diag_normal_log_prob(params, sd_diag=self.prior_std, normalize=False) / getattr(self, "num_data", len(images))
        )
        
        return log_post_val, alpha
    
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
        eval_frequency: int = 20,
        test_loader = None,
        use_wandb: bool = False,
        wandb_project: str = "bnn-training",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict] = None,
        use_lr_schedule: bool = True,
        lr_cycles: Optional[int] = None,
        edl_kl_anneal_epochs: int = 10,
        **kwargs
    ):
        """
        Train the BNN using MCMC sampling.
        
        Args:
            train_loader: Training data loader
            num_epochs: Maximum number of training epochs
            lr: Initial learning rate
            num_burn_in: Number of burn-in epochs
            verbose: Whether to print training progress
            eval_frequency: Frequency of test evaluation (every N epochs after burn-in)
            test_loader: Test data loader for periodic evaluation during training
            use_wandb: Whether to use Weights & Biases for experiment tracking
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_config: Additional config to log to W&B
            use_lr_schedule: Whether to use cyclical cosine learning rate schedule (default: True)
            lr_cycles: Number of cycles for cosine schedule (default: 10, only used if use_lr_schedule=True)
            edl_kl_anneal_epochs: Number of epochs to anneal KL term from 0 to edl_lambda (default: 10)
        """
        # Initialize Weights & Biases if requested
        if use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available. Install with: pip install wandb")
                use_wandb = False
            else:
                # Prepare wandb config
                config = {
                    'num_epochs': num_epochs,
                    'lr': lr,
                    'num_burn_in': num_burn_in,
                    'eval_frequency': eval_frequency,
                    'prior_std': self.prior_std,
                    'temperature': self.temperature,
                    'mcmc_method': self.mcmc_method,
                    'num_parameters': sum(p.numel() for p in self.model.parameters()),
                    'device': str(self.device),
                }
                
                # Add user-provided config
                if wandb_config:
                    config.update(wandb_config)
                
                # Auto-detect experiment directory from wandb_config
                wandb_dir = None
                if wandb_config and 'experiment_dir' in wandb_config:
                    experiment_dir = wandb_config['experiment_dir']
                    wandb_dir = os.path.join(str(experiment_dir), "wandb")
                    os.makedirs(wandb_dir, exist_ok=True)
                
                # Initialize wandb
                wandb_init_kwargs = {
                    'project': wandb_project,
                    'name': wandb_run_name,
                    'config': config,
                    'reinit': True
                }
                if wandb_dir:
                    wandb_init_kwargs['dir'] = wandb_dir
                
                wandb.init(**wandb_init_kwargs)
                
                if verbose:
                    if wandb_dir:
                        print(f"Initialized W&B tracking: project='{wandb_project}', run='{wandb.run.name}', dir='{wandb_dir}'")
                    else:
                        print(f"Initialized W&B tracking: project='{wandb_project}', run='{wandb.run.name}'")
        
        # Store KL annealing configuration
        self.edl_kl_anneal_epochs = edl_kl_anneal_epochs
        if verbose:
            print(f"EDL KL Annealing: 0 → {self.edl_lambda} over {edl_kl_anneal_epochs} epochs")
        
        # Initialize learning rate scheduler with cyclical cosine annealing
        if use_lr_schedule:
            if lr_cycles is None:
                lr_cycles = 10
            
            # Calculate number of samples to collect after burn-in
            # Use n_samples from kwargs if provided, otherwise default to num_epochs - num_burn_in
            n_samples = kwargs.get('n_samples', max(1, num_epochs - num_burn_in))
            
            lr_scheduler = CosineLR(init_lr=lr, n_cycles=lr_cycles, n_samples=n_samples, T_max=num_epochs)
            if verbose:
                print(f"Using Cosine LR: init={lr}, cycles={lr_cycles}, samples={n_samples}, T_max={num_epochs}")
        else:
            lr_scheduler = None
            if verbose:
                print(f"Using constant LR: {lr}")
        
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
        
        # Initialize tracking variables
        self.log_posterior_history = []
        self.training_loss_history = []
        self.test_metrics_history = []
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training BNN", disable=not verbose)
        
        for epoch in epoch_pbar:
            # Set current epoch for KL annealing
            self._current_epoch = epoch
            
            # Update learning rate using cyclical cosine scheduler
            if use_lr_schedule and lr_scheduler:
                current_lr = lr_scheduler.get_lr(epoch)
                # Rebuild transform with new learning rate
                self.build_transform(lr=current_lr, num_data=num_data, **kwargs)
                # Re-initialize state from current parameters
                device_params = {name: param.to(self.device) for name, param in self.state.params.items()} if hasattr(self.state, 'params') else {name: param.to(self.device) for name, param in self.params.items()}
                self.state = self.transform.init(device_params)
                self.state = self._move_state_to_device(self.state)
            else:
                # If no LR schedule, just re-initialize state from current parameters
                device_params = {name: param.to(self.device) for name, param in self.state.params.items()} if hasattr(self.state, 'params') else {name: param.to(self.device) for name, param in self.params.items()}
                self.state = self.transform.init(device_params)
                self.state = self._move_state_to_device(self.state)
            
            num_batches = 0
            epoch_log_posteriors = []
            epoch_losses = []
            
            # Create batch progress bar
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False, disable=not verbose)
            
            for batch in batch_pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Compute log posterior for tracking (before update)
                log_post_val, alpha = self.log_posterior(self.state.params if hasattr(self.state, 'params') else self.state, batch)
                
                # Extract training loss (EDL loss)
                images, labels = batch
                train_loss = evidential_classification(alpha, labels, lamb=self.get_annealed_edl_lambda(epoch))
                
                # Track metrics
                epoch_log_posteriors.append(log_post_val.item())
                epoch_losses.append(train_loss.item())
                
                # Update state
                self.state, info = self.transform.update(self.state, batch)
                
                # Ensure state remains on correct device after update
                self.state = self._move_state_to_device(self.state)
                
                num_batches += 1
                
                # Update batch progress bar
                is_burn_in = epoch < num_burn_in
                batch_pbar.set_postfix({
                    'Batch': f"{num_batches}/{len(train_loader)}",
                    'LogPost': f"{log_post_val.item():.3f}",
                    'Loss': f"{train_loss.item():.3f}",
                    'Samples': len(self.posterior_samples) if not is_burn_in else 0
                })
            
            # Store epoch-level metrics
            avg_log_posterior = np.mean(epoch_log_posteriors)
            avg_train_loss = np.mean(epoch_losses)
            self.log_posterior_history.append(avg_log_posterior)
            self.training_loss_history.append(avg_train_loss)
            
            # Log to wandb
            is_burn_in = epoch < num_burn_in
            if use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    'epoch': epoch + 1,
                    'log_posterior': avg_log_posterior,
                    'train_loss': avg_train_loss,
                    'num_posterior_samples': len(self.posterior_samples),
                    'is_burn_in': is_burn_in,
                    'learning_rate': lr_scheduler.get_lr(epoch) if lr_scheduler else lr,
                    'edl_kl_lambda': self.get_annealed_edl_lambda(epoch)
                }
                wandb.log(log_dict, step=epoch + 1)
            
            # Collect samples after burn-in
            if not is_burn_in:
                # If using LR scheduling, use strategic sampling based on cycles
                if use_lr_schedule and lr_scheduler:
                    if lr_scheduler.should_sample(epoch):
                        sample = self.sample_posterior()
                        self.posterior_samples.append(sample)
                else:
                    # If not using LR scheduling, sample every epoch after burn-in
                    sample = self.sample_posterior()
                    self.posterior_samples.append(sample)
                
                # Periodic test evaluation (start after num_burn_in + eval_frequency to ensure samples are collected)
                first_eval_epoch = num_burn_in + eval_frequency - 1
                if test_loader is not None and epoch >= first_eval_epoch and (epoch - first_eval_epoch) % eval_frequency == 0:
                    if verbose:
                        print(f"\nEvaluating at epoch {epoch+1}...")
                    test_metrics = self.evaluate(test_loader)
                    test_metrics['epoch'] = epoch + 1
                    self.test_metrics_history.append(test_metrics)
                    
                    if verbose:
                        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
                              f"ECE: {test_metrics['ece']:.4f}, "
                              f"Epistemic Unc: {test_metrics['epistemic_uncertainty']:.4f}")
                    
                    # Log test metrics to wandb
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            'test_accuracy': test_metrics['accuracy'],
                            'test_ece': test_metrics['ece'],
                            'test_loss': test_metrics['loss'],
                            'test_total_uncertainty': test_metrics['total_uncertainty'],
                            'test_aleatoric_uncertainty': test_metrics['aleatoric_uncertainty'],
                            'test_epistemic_uncertainty': test_metrics['epistemic_uncertainty'],
                            'test_avg_uncertainty': test_metrics['avg_uncertainty'],
                        }, step=epoch + 1)
            
            # Update epoch progress bar
            burn_in_status = 'Complete' if not is_burn_in else f"{epoch + 1}/{num_burn_in}"
            epoch_pbar.set_postfix({
                'LogPost': f"{avg_log_posterior:.3f}",
                'TrainLoss': f"{avg_train_loss:.3f}",
                'Samples': len(self.posterior_samples),
                'Burn-in': burn_in_status
            })
        
        if verbose:
            print(f"\nTraining completed")
            print(f"Collected {len(self.posterior_samples)} posterior samples.")
        
        # Final evaluation after training completes
        if test_loader is not None:
            if verbose:
                print(f"\nFinal evaluation on test set...")
            final_metrics = self.evaluate(test_loader)
            final_metrics['epoch'] = num_epochs
            self.test_metrics_history.append(final_metrics)
            
            if verbose:
                print(f"Final Test Accuracy: {final_metrics['accuracy']:.4f}, "
                      f"ECE: {final_metrics['ece']:.4f}, "
                      f"Epistemic Unc: {final_metrics['epistemic_uncertainty']:.4f}")
            
            # Log final metrics to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'test_accuracy': final_metrics['accuracy'],
                    'test_ece': final_metrics['ece'],
                    'test_loss': final_metrics['loss'],
                    'test_total_uncertainty': final_metrics['total_uncertainty'],
                    'test_aleatoric_uncertainty': final_metrics['aleatoric_uncertainty'],
                    'test_epistemic_uncertainty': final_metrics['epistemic_uncertainty'],
                    'test_avg_uncertainty': final_metrics['avg_uncertainty'],
                }, step=num_epochs)
        
        # Finish wandb run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
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
        
        # Flatten batch only if the model needs flattened input (MLP)
        if self._needs_flattened_input and len(batch.shape) > 2:
            batch = batch.view(batch.size(0), -1)
        
        # Use all collected samples or a random subset if num_samples is set.
        samples_to_use = self.posterior_samples
        if num_samples is not None and num_samples < len(self.posterior_samples):
            indices = np.random.choice(len(self.posterior_samples), num_samples, replace=False)
            samples_to_use = [self.posterior_samples[i] for i in indices]
        
        predictions = []
        
        for sample_params in samples_to_use:
            with torch.no_grad():
                alpha = func.functional_call(self.model, sample_params, batch)
                # Convert alpha (Dirichlet parameters) to probabilities
                # Expected probability under Dirichlet: p = alpha / sum(alpha)
                sum_alpha = alpha.sum(dim=1, keepdim=True)
                probs = alpha / sum_alpha
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        
        # Predictive mean over posterior samples.
        mean_pred = torch.mean(predictions, dim=0)
        
        # Simple epistemic uncertainty. Variance of class probabilities across samples, summed over classes.
        # this is a variance based uncertainty, different from entropy based MI (total - aleatoric).
        epistemic_uncertainty = torch.var(predictions, dim=0).sum(dim=1)
        
        return mean_pred, epistemic_uncertainty
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        Evaluate the EDL-BNN on test data using all collected posterior samples.
        
        Uses entropy-based uncertainty decomposition (same as StandardBNN for fair comparison):
        - Total: H(E_s[p]) = entropy of expected probabilities
        - Aleatoric: E_s[H(p)] = expected entropy across posterior samples
        - Epistemic: Total - Aleatoric = uncertainty from parameter variation
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self, 'posterior_samples') or len(self.posterior_samples) == 0:
            raise ValueError("No posterior samples available! Train the model first.")
        
        all_predictions = []
        all_labels = []
        all_epistemic = []
        
        # Sample-weighted accumulators
        sum_loss = 0.0
        sum_total_u = 0.0
        sum_aleatoric_u = 0.0
        sum_epistemic_u = 0.0
        total_N = 0
        
        num_samples = len(self.posterior_samples)
        
        # Progress bar over test batches
        eval_pbar = tqdm(test_loader, desc="Evaluating EDL-BNN", leave=False)
        
        eps = 1e-8
        batch_idx = 0
        
        for batch in eval_pbar:
            images, labels = batch
            labels = labels.to(self.device)
            images = images.to(self.device)
            B = labels.size(0)
            
            # Flatten images only if the model needs flattened input (MLP)
            if self._needs_flattened_input and len(images.shape) > 2:
                images = images.view(images.size(0), -1)
            
            # Collect alpha parameters from all posterior samples
            # Shape: [num_samples, batch_size, num_classes]
            all_alphas = []
            for sample_params in self.posterior_samples:
                with torch.no_grad():
                    alpha = func.functional_call(self.model, sample_params, images)
                    all_alphas.append(alpha)
            
            alphas = torch.stack(all_alphas, dim=0)  # [S, B, C]
            
            # Convert alpha to probabilities: p = alpha / sum(alpha)
            probs = alphas / alphas.sum(dim=2, keepdim=True)  # [S, B, C]
            
            # Posterior predictive mean: E_s[p] - average probabilities across samples
            expected_probs = probs.mean(dim=0)  # [B, C]
            expected_probs = torch.clamp(expected_probs, min=eps)
            expected_probs = expected_probs / expected_probs.sum(dim=1, keepdim=True)
            
            # Mean prediction
            all_predictions.append(expected_probs)
            all_labels.append(labels)
            
            # Loss: negative log-likelihood of expected probs
            loss_batch = torch.nn.functional.nll_loss(torch.log(expected_probs), labels, reduction='mean')
            
            # Entropy-based uncertainty decomposition:
            #   H_total = H(E_s[p]) = entropy of expected (averaged) probabilities
            #   H_aleatoric = E_s[H(p)] = expected entropy across posterior samples
            #   H_epistemic = H_total - H_aleatoric = uncertainty due to parameter uncertainty
            
            # Total uncertainty: entropy of the expected probabilities [B]
            total_u = -(expected_probs * torch.log(expected_probs + eps)).sum(dim=1)
            
            # Aleatoric uncertainty: mean entropy per sample across posterior [B]
            ent_per_sample = -(probs * torch.log(probs + eps)).sum(dim=2)  # [S, B]
            aleatoric_u = ent_per_sample.mean(dim=0)  # [B]
            
            # Epistemic uncertainty: difference (should be non-negative)
            epistemic_u = torch.clamp(total_u - aleatoric_u, min=0)  # [B]
            all_epistemic.append(epistemic_u)
            
            # Accumulate metrics
            sum_loss += float(loss_batch.item()) * B
            sum_total_u += total_u.sum().item()
            sum_aleatoric_u += aleatoric_u.sum().item()
            sum_epistemic_u += epistemic_u.sum().item()
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
        all_epistemic = torch.cat(all_epistemic, dim=0)
        
        accuracy = self.compute_accuracy(all_predictions, all_labels)
        ece = self.compute_ece(all_predictions, all_labels)
        avg_uncertainty = all_epistemic.mean().item()
        
        # Final sample-weighted metrics
        loss = sum_loss / max(1, total_N)
        total_uncertainty = sum_total_u / max(1, total_N)
        aleatoric_uncertainty = sum_aleatoric_u / max(1, total_N)
        epistemic_uncertainty = sum_epistemic_u / max(1, total_N)
        
        return {
            'accuracy': accuracy,
            'ece': ece,
            'avg_uncertainty': avg_uncertainty,
            'num_posterior_samples': num_samples,
            'num_test_points': len(all_predictions),
            'loss': loss,
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
        }
