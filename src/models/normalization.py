"""
Filter Response Normalization and Thresholded Linear Unit implementations.

Exactly follows the implementation from:
https://github.com/activatedgeek/understanding-bayesian-classification/blob/main/src/data_aug/nn/filter_response_norm.py
"""

import torch
import torch.nn as nn


__all__ = [
    'FilterResponseNorm2d',
    'TLU2d',
]


class _TLU(nn.Module):
    """
    Thresholded Linear Unit base class.
    
    Implements: max(x, tau) where tau is a learnable parameter.
    """
    def __init__(self, num_features, num_dims):
        super().__init__()

        self.num_features = num_features
        self.num_dims = num_dims

        self.tau = nn.Parameter(torch.zeros([1, num_features] + [1] * num_dims))

    def forward(self, inputs):
        return torch.max(inputs, self.tau)


class TLU2d(_TLU):
    """
    Thresholded Linear Unit for 2D inputs.
    
    Expects inputs of shape (B x num_features x H x W)
    """
    def __init__(self, num_features):
        super().__init__(num_features, 2)


class _FilterResponseNorm(nn.Module):
    """
    Filter Response Normalization base class.
    
    Normalizes feature maps across specified dimensions.
    """
    def __init__(self, num_features, avg_dims, eps=1e-6, learnable_eps=False):
        super().__init__()

        self.num_features = num_features
        self.avg_dims = avg_dims

        self.gamma = nn.Parameter(torch.ones([1, num_features] + [1] * len(avg_dims)))
        self.beta = nn.Parameter(torch.zeros([1, num_features] + [1] * len(avg_dims)))
        if learnable_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))

    def forward(self, inputs):
        nu2 = (inputs**2).mean(dim=self.avg_dims, keepdim=True)

        x = inputs * (nu2 + self.eps.abs()).rsqrt()

        return self.gamma * x + self.beta


class FilterResponseNorm2d(_FilterResponseNorm):
    """
    Filter Response Normalization for 2D inputs.
    
    Expects inputs of shape (B x num_features x H x W)
    
    Normalizes across spatial dimensions (height and width).
    """
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, [-2, -1], **kwargs)
