"""
ResNet20 for CIFAR-10 with Filter Response Normalization.
Adapted from https://github.com/activatedgeek/understanding-bayesian-classification/
"""

import torch
import torch.nn as nn
from src.registry import MODEL_REGISTRY


class _FilterResponseNorm(nn.Module):
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
    '''
    Expects inputs of shape (B x num_features x H x W)
    '''
    def __init__(self, num_features, **kwargs):
        super().__init__(num_features, [-2, -1], **kwargs)


class _TLU(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()

        self.num_features = num_features
        self.num_dims = num_dims

        self.tau = nn.Parameter(torch.zeros([1, num_features] + [1] * num_dims))

    def forward(self, inputs):
        return torch.max(inputs, self.tau)


class TLU2d(_TLU):
    def __init__(self, num_features):
        super().__init__(num_features, 2)



class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = FilterResponseNorm2d(planes)
        self.tlu1 = TLU2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = FilterResponseNorm2d(planes)
        self.tlu2 = TLU2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                FilterResponseNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.tlu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.tlu2(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = FilterResponseNorm2d(16)
        self.tlu1 = TLU2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        # self.pool = nn.AvgPool2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tlu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


@MODEL_REGISTRY.register("resnet20")
class ResNet20(_ResNet):
    """
    ResNet20 for CIFAR-10 with Filter Response Normalization.
    ~273k parameters.
    """
    def __init__(self, num_classes=10):
         super().__init__(_BasicBlock, [3, 3, 3], num_classes)


