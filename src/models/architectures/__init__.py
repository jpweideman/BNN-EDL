"""
Model architectures with automatic registration.

All models in this package are automatically imported and registered
when this package is imported.
"""

import importlib
import pkgutil

# Automatically import all modules in this package to trigger @register decorators
for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")

