#!/usr/bin/env python3
"""
Utility script to run training with configuration files.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path


def load_and_run_config(config_path: str, script_name: str, resume_from: str = None):
    """Load config file and run training script with those parameters."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build command line arguments
    cmd = [sys.executable, script_name]
    
    # List arguments that need special handling
    list_args = {
        'mlp_hidden_sizes', 'cnn_conv_channels', 'cnn_fc_hidden_sizes'
    }
    
    # Boolean arguments that need special handling
    bool_args = {
        'cnn_use_batch_norm', 'quick', 'no_periodic_eval', 'use_wandb', 'use_adaptive_burnin', 'use_lr_schedule'
    }
    
    for key, value in config.items():
        # Skip null values
        if value is None:
            continue
            
        if key in list_args and isinstance(value, list):
            # For list arguments, add each element separately
            if value:  # Only add if list is not empty
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])
        elif key in bool_args:
            # For boolean arguments, add the appropriate flag
            if key == 'use_lr_schedule':
                # For use_lr_schedule, pass the value as true/false
                cmd.extend([f"--{key}", str(value).lower()])
            elif value:
                # For other boolean args, only add the flag if True
                cmd.append(f"--{key}")
        else:
            # For regular arguments
            cmd.extend([f"--{key}", str(value)])
    
    # Add resume_from argument if specified
    if resume_from:
        # Convert to absolute path if it's relative
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            # Get project root (parent of training directory)
            project_root = Path(__file__).parent.parent
            resume_path = project_root / resume_path
        cmd.extend(["--resume_from", str(resume_path)])
    
    print(f"Running: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the training script from the training directory
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run training with config file")
    parser.add_argument("config", type=str, help="Path to config JSON file")
    parser.add_argument("--script", type=str, choices=['train_bnn.py', 'train_edl_bnn.py', 'train_baseline.py', 'train_edl.py'], 
                       required=True, help="Which training script to use")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to experiment directory to resume from (for train_baseline.py only)")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} not found!")
        return 1
    
    return load_and_run_config(args.config, args.script, resume_from=args.resume_from)


if __name__ == "__main__":
    exit(main())
