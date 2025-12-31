"""Multi-Layer Perceptron (MLP) architecture."""

import torch.nn as nn
from src.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("mlp")
class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron with ReLU activations.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability (0 means no dropout)
        """
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)

