"""Likelihood functions for Bayesian inference."""

import pkgutil
import importlib

# Auto-import all likelihood modules
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f'{__name__}.{module_name}')

