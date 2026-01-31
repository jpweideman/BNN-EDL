"""Multi-Layer Perceptron (MLP) architecture."""

import torch.nn as nn
from src.registry import MODEL_REGISTRY
from src.builders.output_layer_builder import OutputLayerBuilder


@MODEL_REGISTRY.register("mlp")
class MLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron with ReLU activations.
    """
    
    def __init__(self, input_dim, hidden_dims, output_layer_config, dropout=0.0):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_layer_config: Output layer configuration dict
            dropout: Dropout probability (0 means no dropout)
        """
        super().__init__()
        
        # Build backbone 
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.backbone = nn.Sequential(*layers)
        
        # Build output layer 
        self.output_layer = OutputLayerBuilder(output_layer_config).build(
            input_dim=hidden_dims[-1]
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        return self.output_layer(features)
