"""
Base BNN class with common functionality.
"""

import torch
import torch.nn as nn
from torch import func
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
from abc import ABC, abstractmethod


class BaseBNN(ABC):
    """
    Base class for Bayesian Neural Networks.
    Provides common functionality for different BNN implementations.
    """
    
    def __init__(self, model: nn.Module, num_classes: int, device: str = 'auto'):
        """
        Initialize the BNN.
        
        Args:
            model: PyTorch model architecture
            num_classes: Number of output classes
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Device management
        self.device = self._setup_device(device)
        
        # Move model to device
        self.model = model.to(self.device)
        self.num_classes = num_classes
        # Get parameters after moving to device
        self.params = dict(self.model.named_parameters())
        self.state = None
        
        print(f"BNN initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """
        Setup and validate device.
        
        Args:
            device: Device specification ('auto', 'cuda', 'cpu')
            
        Returns:
            torch.device object
        """
        if device == 'auto':
            if torch.cuda.is_available():
                selected_device = torch.device('cuda')
                print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            else:
                selected_device = torch.device('cpu')
                print("No GPU available, using CPU")
        elif device == 'cuda':
            if torch.cuda.is_available():
                selected_device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA requested but not available, falling back to CPU")
                selected_device = torch.device('cpu')
        elif device == 'cpu':
            selected_device = torch.device('cpu')
            print("Using CPU")
        else:
            raise ValueError(f"Invalid device: {device}. Use 'auto', 'cuda', or 'cpu'")
        
        return selected_device
    
    def _move_batch_to_device(self, batch):
        """
        Move batch data to the correct device.
        
        Args:
            batch: Batch data (images, labels)
            
        Returns:
            Batch moved to device
        """
        if isinstance(batch, (tuple, list)):
            return tuple(item.to(self.device) if torch.is_tensor(item) else item for item in batch)
        elif torch.is_tensor(batch):
            return batch.to(self.device)
        else:
            return batch
        
    @abstractmethod
    def build_transform(self, **kwargs):
        """Build the posteriors transform for this BNN type."""
        pass
    
    @abstractmethod
    def log_posterior(self, params: Dict[str, torch.Tensor], batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log posterior probability.
        
        Args:
            params: Model parameters
            batch: (images, labels) batch
            
        Returns:
            Tuple of (log_posterior_value, model_output)
        """
        pass
    
    @abstractmethod
    def predict_batch(self, batch: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for a batch with uncertainty quantification.
        
        Args:
            batch: Input batch
            num_samples: Number of posterior samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        pass
    
    @abstractmethod
    def sample_posterior(self) -> Dict[str, torch.Tensor]:
        """Sample parameters from the current posterior state."""
        pass
    
    @abstractmethod
    def fit(self, train_loader, num_epochs: int = 100, lr: float = 1e-3, num_burn_in: int = 50, **kwargs):
        """
        Train the BNN.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            num_burn_in: Number of burn-in epochs
            **kwargs: Additional training arguments
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_loader, **kwargs) -> Dict[str, float]:
        """
        Evaluate the BNN on test data.
        
        Args:
            test_loader: Test data loader
            **kwargs: BNN-specific evaluation arguments
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute classification accuracy."""
        pred_classes = torch.argmax(predictions, dim=1)
        return (pred_classes == labels).float().mean().item()
    
    def compute_ece(self, predictions: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities [batch_size, num_classes]
            labels: True labels [batch_size]
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value
        """
        confidences = torch.max(predictions, dim=1)[0]
        pred_classes = torch.argmax(predictions, dim=1)
        accuracies = (pred_classes == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()


def create_mlp(input_size: int, hidden_sizes: list, num_classes: int, dropout_rate: float = 0.1) -> nn.Module:
    """
    Create a Multi-Layer Perceptron for classification.
    
    Args:
        input_size: Size of input features
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        PyTorch MLP model
    """
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    
    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], num_classes))
    
    return nn.Sequential(*layers)


def create_cnn(
    input_shape: Tuple[int, int, int], 
    num_classes: int, 
    dropout_rate: float = 0.25,
    conv_channels: Optional[list] = None,
    conv_layers_per_block: int = 2,
    fc_hidden_sizes: Optional[list] = None,
    kernel_size: int = 3,
    pool_size: int = 2,
    use_batch_norm: bool = False
) -> nn.Module:
    """
    Create a configurable CNN for classification.
    
    Args:
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        conv_channels: List of channel sizes for conv blocks [32, 64, 128]. If None, uses [32, 64]
        conv_layers_per_block: Number of conv layers per block (default: 2)
        fc_hidden_sizes: List of fully connected layer sizes. If None, uses [512]
        kernel_size: Kernel size for convolutions (default: 3)
        pool_size: Pool size for max pooling (default: 2)
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        PyTorch CNN model
    """
    channels, height, width = input_shape
    
    # Default configurations
    if conv_channels is None:
        conv_channels = [32, 64]
    if fc_hidden_sizes is None:
        fc_hidden_sizes = [512]
    
    layers = []
    in_channels = channels
    current_height, current_width = height, width
    
    # Build convolutional blocks
    for block_idx, out_channels in enumerate(conv_channels):
        # Add conv layers for this block
        for layer_idx in range(conv_layers_per_block):
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size, padding=kernel_size//2))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        # Add pooling and dropout after each block
        layers.append(nn.MaxPool2d(pool_size))
        layers.append(nn.Dropout(dropout_rate))
        
        # Update spatial dimensions
        current_height //= pool_size
        current_width //= pool_size
    
    # Flatten for fully connected layers
    layers.append(nn.Flatten())
    
    # Calculate flattened size
    flattened_size = conv_channels[-1] * current_height * current_width
    
    # Build fully connected layers
    fc_input_size = flattened_size
    for fc_size in fc_hidden_sizes:
        layers.append(nn.Linear(fc_input_size, fc_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 2))  # Higher dropout in dense layers
        fc_input_size = fc_size
    
    # Output layer
    layers.append(nn.Linear(fc_input_size, num_classes))
    
    return nn.Sequential(*layers)


def _weights_init(m):
    """
    Initialize weights using Kaiming normal initialization.
    
    Reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """Lambda layer for functional operations."""
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class FunctionalBatchNorm2d(nn.Module):
    """
    Simplified BatchNorm2d that works with torch.func.functional_call().
    
    This implementation uses only batch statistics during training to avoid
    issues with running statistics in functional calls.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(FunctionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            # These are learnable parameters
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
    def forward(self, input):
        if self.training:
            # Always use batch statistics during training for BNN compatibility
            batch_mean = input.mean([0, 2, 3])
            batch_var = input.var([0, 2, 3], unbiased=False)
            mean = batch_mean
            var = batch_var
        else:
            # During evaluation, also use batch statistics for simplicity
            # This ensures consistent behavior in functional calls
            batch_mean = input.mean([0, 2, 3])
            batch_var = input.var([0, 2, 3], unbiased=False)
            mean = batch_mean
            var = batch_var
        
        # Normalize
        input_normalized = (input - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            input_normalized = input_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return input_normalized


class BasicBlock(nn.Module):
    """
    Basic residual block for CIFAR-10 ResNet.
    
    Modified from the original implementation to be compatible with 
    functional calls by using FunctionalBatchNorm2d instead of standard BatchNorm.
    
    Reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        # Use bias=False to match reference implementation exactly
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = FunctionalBatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = FunctionalBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                Modified to work without BatchNorm for functional compatibility.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            torch.nn.functional.pad(x[:, :, ::2, ::2], 
                                                                   (0, 0, 0, 0, planes//4, planes//4), 
                                                                   "constant", 0))
            elif option == 'B':
                # Option B to match reference implementation
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     FunctionalBatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # Match reference implementation exactly with BatchNorm
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """
    Properly implemented ResNet for CIFAR-10 as described in the original paper.
    
    Modified from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    Author: Yerlan Idelbayev
    
    This implementation is adapted for BNN functional calls by removing BatchNorm
    layers and using bias=True in convolutions.
    
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # Match reference implementation exactly with BatchNorm
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = FunctionalBatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # Match reference implementation exactly with BatchNorm
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_resnet20(input_shape: Tuple[int, int, int], num_classes: int) -> nn.Module:
    """
    Create a ResNet-20 model for CIFAR-10.
    
    Properly implemented ResNet-20 as described in the original paper.
    Uses FunctionalBatchNorm2d for BNN compatibility with torch.func.functional_call().
    
    Reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    Author: Yerlan Idelbayev
    
    Args:
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        ResNet-20 model with ~0.27M parameters and functional BatchNorm
    """
    channels, height, width = input_shape
    return ResNet(BasicBlock, [3, 3, 3], num_classes, input_channels=channels)


