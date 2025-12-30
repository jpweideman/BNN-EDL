"""Neural network models."""

# Auto-import all architecture modules to trigger registration
import importlib
import pkgutil

# Import architectures subpackage
from . import architectures

__all__ = ['architectures']

