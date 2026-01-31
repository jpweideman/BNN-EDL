"""Evidential Dirichlet output layer."""

import torch.nn as nn
from edl_pytorch import Dirichlet
from src.registry import OUTPUT_LAYER_REGISTRY


@OUTPUT_LAYER_REGISTRY.register("dirichlet")
class DirichletOutput(nn.Module):
    """Evidential output layer using Dirichlet distribution.
    
    Outputs alpha parameters for each class.
    """
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dirichlet = Dirichlet(in_features=input_dim, out_units=num_classes)
    
    def forward(self, x):
        return self.dirichlet(x)
