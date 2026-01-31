"""Standard linear output layer."""

import torch.nn as nn
from src.registry import OUTPUT_LAYER_REGISTRY


@OUTPUT_LAYER_REGISTRY.register("linear")
class LinearOutput(nn.Module):
    """Standard linear classification head."""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)
