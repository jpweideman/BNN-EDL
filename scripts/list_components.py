"""List all registered components with their parameters."""

import sys
from pathlib import Path
import inspect

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import to trigger registrations
import src.data.transforms

from src.registry import (
    TRANSFORM_REGISTRY,
    MODEL_REGISTRY,
    LOSS_REGISTRY,
    METRIC_REGISTRY,
    SAMPLER_REGISTRY,
    DATASET_REGISTRY
)


def print_component_info(registry, title):
    """Print information about all components in a registry."""
    print(f"\n{title}")
    print(f"{'-'*len(title)}\n")
    
    components = list(registry._registry.keys())
    
    if not components:
        print("  No components registered yet.\n")
        return
    
    for name in sorted(components):
        cls = registry._registry[name]
        
        # Get class docstring
        doc = inspect.getdoc(cls) or "No description available."
        
        # Get __init__ signature
        try:
            sig = inspect.signature(cls.__init__)
            params = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Get parameter info
                if param.default != inspect.Parameter.empty:
                    params.append(f"{param_name}={param.default}")
                else:
                    params.append(param_name)
            
            params_str = ", ".join(params) if params else "no parameters"
        except Exception:
            params_str = "unknown parameters"
        
        print(f"  {name}")
        print(f"    Description: {doc}")
        print(f"    Parameters: {params_str}")
        print()


def main():
    """List all registered components."""
    print("\nRegistered Components")
    print("-"*len("Registered Components\n"))
    
    print_component_info(TRANSFORM_REGISTRY, "Transforms")
    print_component_info(MODEL_REGISTRY, "Models")
    print_component_info(LOSS_REGISTRY, "Losses")
    print_component_info(METRIC_REGISTRY, "Metrics")
    print_component_info(SAMPLER_REGISTRY, "Samplers")
    print_component_info(DATASET_REGISTRY, "Datasets")


if __name__ == "__main__":
    main()

