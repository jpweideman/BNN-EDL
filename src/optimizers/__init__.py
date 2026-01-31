"""Optimizers with automatic registration."""

import importlib
import pkgutil
from pathlib import Path

# Get the path to this package
package_path = Path(__file__).parent

# Import standard optimizers
standard_path = package_path / "standard"
if standard_path.exists():
    for _, module_name, _ in pkgutil.iter_modules([str(standard_path)]):
        importlib.import_module(f"{__name__}.standard.{module_name}")

# Import BNN optimizers, excluding base and utils
bnn_path = package_path / "bnn"
if bnn_path.exists():
    for _, module_name, _ in pkgutil.iter_modules([str(bnn_path)]):
        if module_name not in ['base', 'utils']:
            importlib.import_module(f"{__name__}.bnn.{module_name}")