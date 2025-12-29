# Bayesian Neural Networks with Evidential Deep Learning

This repository implements Bayesian Neural Networks (BNNs) using the [posteriors](https://github.com/normal-computing/posteriors) library for uncertainty quantification in deep learning.

## Current Implementation

### ✅ Standard BNN with MCMC Sampling

- **MCMC Methods**: SGLD (Stochastic Gradient Langevin Dynamics), SGHMC (Stochastic Gradient Hamiltonian Monte Carlo)
- **Architectures**: Multi-Layer Perceptrons (MLP), Convolutional Neural Networks (CNN), ResNet-20
- **Datasets**: MNIST, CIFAR-10
- **Uncertainty**: Epistemic uncertainty through parameter sampling

### 🚧 Future Implementations

- **EDL BNN**: Evidential Deep Learning with MCMC sampling (files prepared in `src/losses/`)
- **Hybrid Models**: Combining EDL with traditional BNN approaches
- **Advanced Architectures**: ResNet, Transformer-based models

## Repository Structure

```
bnn-edl/
├── src/
│   ├── models/
│   │   ├── standard_bnn.py      # Standard BNN with posteriors MCMC
│   │   ├── base_bnn.py          # Base classes and model architectures
│   │   └── __init__.py
│   ├── datasets/
│   │   ├── classification.py    # MNIST, CIFAR-10 data loaders
│   │   └── __init__.py
│   ├── utils/
│   │   ├── training.py          # Training utilities and model comparison
│   │   ├── evaluation.py        # Evaluation metrics and uncertainty analysis
│   │   └── __init__.py
│   └── losses/
│       ├── edl_loss.py          # EDL loss functions (for future use)
│       └── __init__.py
├── notebooks/
│   ├── standard_bnn_demo.ipynb  # Comprehensive demo notebook
│   └── 01_standard_bnn_demo.ipynb  # Simple demo (from user testing)
├── data/                        # Dataset storage
├── experiments/                 # Experiment results
├── results/                     # Output results
├── requirements.txt             # Dependencies
├── test_bnn.py                  # Quick test script
└── README.md
```

## Installation

### Using Poetry (Recommended)

This project requires **Python 3.10** and uses Poetry for dependency management.

1. **Install Python 3.10** (if not already installed):
   ```bash
   # macOS (Homebrew)
   brew install python@3.10
   
   # Ubuntu/Debian
   sudo apt install python3.10
   
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bnn-edl
   ```

4. **Set Python 3.10 and install all dependencies**:
   ```bash
   poetry env use python3.10
   poetry install
   ```

5. **Activate the virtual environment**:
   ```bash
   source $(poetry env info --path)/bin/activate
   ```

6. **Run training**:
   ```bash
   # After activation, run commands normally
   python training/load_config.py training/configs/mnist_mlp.json
   
   # Or use poetry run without activation
   poetry run python training/load_config.py training/configs/mnist_mlp.json
   ```

## Usage

### Basic Example

```python
from src.datasets.classification import get_mnist_dataloaders, get_dataset_info
from src.models.standard_bnn import StandardBNN
from src.models.base_bnn import create_mlp

# Load data
train_loader, val_loader, test_loader = get_mnist_dataloaders(
    data_dir="./data", batch_size=128, flatten=True
)
from datasets import infer_dataset_info
dataset_info = infer_dataset_info(train_loader)

# Create model
model = create_mlp(
    input_size=dataset_info['input_size'],
    hidden_sizes=[256, 128],
    num_classes=dataset_info['num_classes']
)

# Create BNN with device options
bnn = StandardBNN(
    model=model,
    num_classes=dataset_info['num_classes'],
    mcmc_method="sgld",
    device='auto'  # 'auto', 'cuda', or 'cpu'
)

# Train
bnn.fit(train_loader, num_epochs=50, lr=1e-3, num_burn_in=20)

# Predict with uncertainty
predictions, uncertainties = bnn.predict_batch(test_images, num_samples=100)

# Evaluate
metrics = bnn.evaluate(test_loader, num_samples=100)
```

### Device Options

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Device options:
bnn = StandardBNN(model, num_classes=10, device='auto')   # Auto-detect (recommended)
bnn = StandardBNN(model, num_classes=10, device='cuda')   # Force GPU
bnn = StandardBNN(model, num_classes=10, device='cpu')    # Force CPU
```

### Notebooks

- **`notebooks/standard_bnn_demo.ipynb`**: Comprehensive demonstration with MLP and CNN on MNIST and CIFAR-10
- **`notebooks/01_standard_bnn_demo.ipynb`**: Simple MNIST example

## Features

### Uncertainty Quantification
- **Epistemic Uncertainty**: Model uncertainty through parameter sampling
- **Calibration Analysis**: Reliability diagrams and Expected Calibration Error (ECE)
- **Uncertainty Visualization**: Correlation between uncertainty and prediction errors

### Model Architectures
- **MLP**: Fully-connected networks for flattened images
- **CNN**: Convolutional networks for 2D image data
- **ResNet-20**: Properly implemented ResNet-20 for CIFAR-10 with residual connections

### MCMC Methods
- **SGLD**: Stochastic Gradient Langevin Dynamics - Basic MCMC with Langevin noise
- **SGHMC**: Stochastic Gradient Hamiltonian Monte Carlo - Uses momentum for better mixing
- **SGNHT**: Stochastic Gradient Nosé-Hoover Thermostat - Advanced thermostat control
- **BAOA**: Bayesian Averaging Over Architectures - Architecture uncertainty

### Device Support
- **Auto-Detection**: Automatically uses GPU if available, falls back to CPU
- **Manual Override**: Force CPU or GPU usage with `device` parameter
- **Memory Management**: Efficient GPU memory handling for large models

### Evaluation Metrics
- **Accuracy**: Classification accuracy
- **ECE**: Expected Calibration Error
- **Uncertainty Statistics**: Mean epistemic uncertainty
- **Calibration Plots**: Reliability diagrams

## Dependencies

- `torch>=2.0.0`: PyTorch for deep learning
- `posteriors>=0.1.0`: Bayesian inference library
- `torchvision>=0.15.0`: Computer vision datasets and transforms
- `numpy>=1.20.0`: Numerical computing
- `matplotlib>=3.5.0`: Plotting
- `seaborn>=0.11.0`: Statistical visualization
- `scikit-learn>=1.0.0`: Machine learning utilities
- `tqdm>=4.60.0`: Progress bars



