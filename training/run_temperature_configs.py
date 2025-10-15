#!/usr/bin/env python3
"""
Simple temperature study using config files.

This script creates config files for different temperatures and runs them.
"""

import json
import subprocess
import sys
from pathlib import Path


def create_temperature_configs():
    """Create config files for different temperature values."""
    
    # Base configuration for CIFAR-10 ResNet20
    base_config = {
        "dataset": "cifar10",
        "architecture": "resnet20",
        "batch_size": 128,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "num_burn_in": 25,
        "prior_std": 1.0,
        "mcmc_method": "sgld",
        "mcmc_beta": 0.0,
        "mcmc_alpha": 0.01,
        "mcmc_sigma": 1.0,
        "mcmc_xi": 0.01,
        "mcmc_momenta": None,
        "device": "auto",
        "num_workers": 2,
        "seed": 123
    }
    
    temperatures = [0.03, 0.1, 0.3, 1.0, 3.0]
    
    config_dir = Path(__file__).parent / "configs"
    config_files = []
    
    for temp in temperatures:
        config = base_config.copy()
        config["temperature"] = temp
        config["experiment_name"] = f"cifar10_resnet20_temp_{temp}"
        
        # Create filename
        config_file = config_dir / f"cifar10_resnet20_temp_{temp}.json"
        
        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        config_files.append(config_file)
        print(f"Created config: {config_file}")
    
    return config_files


def run_temperature_study():
    """Run all temperature experiments."""
    
    print("Creating temperature study configs...")
    config_files = create_temperature_configs()
    
    print(f"\n🚀 Running {len(config_files)} temperature experiments...")
    
    script_dir = Path(__file__).parent
    results = []
    
    for i, config_file in enumerate(config_files, 1):
        temp = config_file.stem.split('_')[-1]  # Extract temperature from filename
        print(f"\n[{i}/{len(config_files)}] Running temperature = {temp}")
        print(f"Config: {config_file}")
        
        # Run using load_config.py
        cmd = [sys.executable, "load_config.py", str(config_file)]
        
        try:
            result = subprocess.run(cmd, cwd=script_dir)
            
            if result.returncode == 0:
                print(f"Temperature {temp} completed successfully")
                results.append({"temperature": temp, "status": "success"})
            else:
                print(f"Temperature {temp} failed with return code {result.returncode}")
                results.append({"temperature": temp, "status": "failed"})
                
        except Exception as e:
            print(f"Temperature {temp} failed with exception: {e}")
            results.append({"temperature": temp, "status": "exception", "error": str(e)})
    
    # Print summary
    print(f"\n🎯 Temperature Study Complete!")
    print("\nResults Summary:")
    print("-" * 40)
    print(f"{'Temperature':<12} {'Status':<10}")
    print("-" * 40)
    
    successful = 0
    for result in results:
        temp = result["temperature"]
        status = result["status"]
        print(f"{temp:<12} {status:<10}")
        if status == "success":
            successful += 1
    
    print(f"\n{successful}/{len(results)} experiments completed successfully!")
    
    return results


if __name__ == "__main__":
    run_temperature_study()
