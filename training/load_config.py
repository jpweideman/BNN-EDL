#!/usr/bin/env python3
"""
Utility script to run training with configuration files.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path


def load_and_run_config(config_path: str):
    """Load config file and run training script with those parameters."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build command line arguments
    cmd = [sys.executable, "train_bnn.py"]
    
    # List arguments that need special handling
    list_args = {
        'mlp_hidden_sizes', 'cnn_conv_channels', 'cnn_fc_hidden_sizes'
    }
    
    # Boolean arguments that need special handling
    bool_args = {
        'cnn_use_batch_norm', 'quick', 'augment'
    }
    
    for key, value in config.items():
        if value is None:
            # Skip None/null values - don't pass them as arguments
            continue
        elif key in list_args and isinstance(value, list):
            # For list arguments, add each element separately
            if value:  # Only add if list is not empty
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])
        elif key in bool_args:
            # For boolean arguments, only add the flag if True
            if value:
                cmd.append(f"--{key}")
        else:
            # For regular arguments
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Run the training script
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run BNN training with config file")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} not found!")
        return 1
    
    return load_and_run_config(args.config)


if __name__ == "__main__":
    exit(main())
