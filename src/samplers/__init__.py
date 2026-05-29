"""MCMC samplers with automatic registration."""

import pkgutil
import importlib

for _, module_name, _ in pkgutil.iter_modules(__path__):
    if module_name != 'base':
        importlib.import_module(f'{__name__}.{module_name}')
