# BNN Training System

This directory contains the training infrastructure for Bayesian Neural Networks.

## Files

- `train_bnn.py` - Main training script
- `load_config.py` - Utility to run training with config files
- `configs/` - Example configuration files

## Quick Start

### 1. Basic Training

Run a quick test to make sure everything works:

```bash
cd training
python train_bnn.py --quick --dataset mnist --architecture mlp
```

### 2. Full Training Examples

Train MNIST with MLP:
```bash
python train_bnn.py --dataset mnist --architecture mlp --num_epochs 100
```

Train CIFAR-10 with CNN:
```bash
python train_bnn.py --dataset cifar10 --architecture cnn --num_epochs 150 --batch_size 64
```

### 3. Using Configuration Files

Use pre-made configs:
```bash
python load_config.py configs/mnist_mlp.json
python load_config.py configs/cifar10_cnn.json
python load_config.py configs/quick_test.json
```

## Command Line Arguments

### Dataset & Model
- `--dataset` - Dataset to use (mnist, cifar10)
- `--architecture` - Model architecture (mlp, cnn, resnet20)

### Training Parameters
- `--batch_size` - Batch size (default: 128)
- `--num_epochs` - Number of training epochs (default: 100)
- `--learning_rate` - Learning rate (default: 1e-3)
- `--num_burn_in` - Number of burn-in epochs (default: 50)

### BNN Parameters
- `--prior_std` - Prior standard deviation (default: 1.0)
- `--temperature` - Temperature for posterior (default: 1.0)
- `--mcmc_method` - MCMC method (sgld, sghmc, sgnht, baoa)
- `--mcmc_beta` - Gradient noise coefficient for all methods (default: 0.0)
- `--mcmc_alpha` - Friction coefficient for SGHMC/SGNHT/BAOA (default: 0.01)
- `--mcmc_sigma` - Standard deviation of momenta target distribution for SGHMC/SGNHT/BAOA (default: 1.0)
- `--mcmc_xi` - Initial thermostat value for SGNHT (default: alpha)
- `--mcmc_momenta` - Initial momenta for SGHMC/SGNHT/BAOA (default: None for random initialization)

### System Parameters
- `--device` - Device to use (auto, cuda, cpu)
- `--num_workers` - Number of data loader workers (default: 0)

### Experiment Management
- `--experiment_name` - Name for the experiment
- `--output_dir` - Directory to save experiments (default: ./experiments)

### Quick Options
- `--quick` - Quick training with reduced epochs for testing

## Output Structure

Each experiment creates a timestamped directory in `experiments/` containing:

```
experiments/
└── experiment_name_20241010_143022/
    ├── config.json      # Experiment configuration
    ├── model.pt         # Saved model and posterior samples
    └── results.json     # Training and evaluation results
```

## Results Format

The `results.json` contains:
- `dataset_info` - Dataset information (shape, classes, etc.)
- `test_metrics` - Test set performance
- `training_config` - Full training configuration

Note: No validation split is used - all training data is used for MCMC sampling as per BNN best practices.

Metrics include:
- `accuracy` - Classification accuracy
- `ece` - Expected Calibration Error
- `avg_uncertainty` - Average epistemic uncertainty (variance-based)
- `total_uncertainty` - Total predictive uncertainty (entropy-based)
- `aleatoric_uncertainty` - Aleatoric uncertainty
- `epistemic_uncertainty` - Epistemic uncertainty (entropy decomposition)
- `num_samples` - Number of posterior samples collected
- `loss` - Negative log-likelihood

## Model Loading

To load a trained model:

```python
from training.train_bnn import load_model
from models.base_bnn import create_mlp

# Create the same architecture
model = create_mlp(784, 10, [128, 64])

# Load the trained BNN
bnn = load_model("experiments/my_experiment_20241010_143022/model.pt", model)

# Use for prediction
predictions, uncertainty = bnn.predict_batch(test_batch)
```

## Configuration Files

Create custom configs by copying and modifying the examples in `configs/`:

```json
{
  "dataset": "mnist",
  "architecture": "mlp",
  "batch_size": 128,
  "num_epochs": 100,
  "learning_rate": 0.001,
  "num_burn_in": 50,
  "prior_std": 1.0,
  "temperature": 1.0,
  "mcmc_method": "sgld",
  "device": "auto",
  "num_workers": 0,
  "experiment_name": "my_experiment"
}
```

## Tips

1. **Start with `--quick`** to test your setup before long runs
2. **Use GPU** for faster training: `--device cuda`
3. **Adjust batch size** based on your GPU memory
4. **Increase burn-in** for better posterior approximation
5. **Monitor uncertainty metrics** to assess model quality
6. **No validation needed** - BNN training uses all available data for better posterior approximation

## Extending

To add new models:
1. Implement your BNN class inheriting from `BaseBNN`
2. Add it to the imports in `train_bnn.py`
3. Add model selection logic in the training script

To add new datasets:
1. Implement dataloader function in `src/datasets/classification.py`
2. Add it to `get_dataloaders()` in `train_bnn.py`
