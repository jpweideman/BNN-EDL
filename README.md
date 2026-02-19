# Bayesian Neural Networks for Classification

This repository implements Bayesian Neural Networks (BNNs) using the [posteriors](https://github.com/normal-computing/posteriors) library and Evidential Deep Learning (EDL) using [edl-pytorch](https://github.com/teddykoker/evidential-learning-pytorch) for uncertainty quantification.



## Features

### Uncertainty Quantification Methods

- **BNN (Bayesian Neural Networks)**: SGLD and SGHMC sampling
- **EDL (Evidential Deep Learning)**: Dirichlet-based uncertainty
- **Hybrid**: Combine BNN and EDL

### Model Architectures

- **MLP**: Multi-Layer Perceptron with configurable hidden dimensions
- **ResNet20**: ResNet-20 with Filter Response Normalization for CIFAR-10

### Output Layers 

- **Linear**: Standard softmax classification
- **Dirichlet**: Evidential output for EDL uncertainty

### Datasets

- **MNIST**: Handwritten digits (28x28)
- **CIFAR-10**: Natural images (32x32x3)

### Uncertainty Metrics

- **Predictive Entropy**: Total uncertainty
- **Expected Entropy**: Aleatoric (data) uncertainty
- **Mutual Information**: Epistemic (model) uncertainty
- **Predictive Variance**: Ensemble disagreement
- **Calibration Error**: ECE (Expected Calibration Error)
- **BMA Accuracy/Loss**: Bayesian Model Averaging metrics

## Modular Configuration System

All components can be **mixed and matched via YAML configs** in `src/configs/`

**Training Modes:**
- Standard + Linear (deterministic baseline)
- Standard + EDL (evidential uncertainty)
- BNN + Linear (Bayesian uncertainty via SGLD/SGHMC)
- BNN + EDL (combined evidential and Bayesian uncertainty)



## Repository Structure

```
BNN-EDL/
├── src/
│   ├── models/
│   │   ├── architectures/       # Model backbones (MLP, ResNet20)
│   │   └── output_layers/       # Output layers (Linear, Dirichlet)
│   ├── optimizers/
│   │   ├── standard/            # Adam, AdamW, SGD
│   │   └── bnn/                 # SGLD, SGHMC + base class
│   ├── losses/                  # Cross-entropy, evidential_classification
│   ├── likelihoods/             # Categorical likelihood for BNN
│   ├── priors/                  # Diagonal normal prior for BNN
│   ├── metrics/                 # Accuracy, uncertainty metrics, calibration
│   ├── data/                    # Datasets (MNIST, CIFAR-10) and transforms
│   ├── training/                # Training engines (standard, BNN) and handlers
│   ├── builders/                # Component builders (model, optimizer, loss, etc.)
│   ├── setup/                   # Setup orchestrators (data, trainer, evaluator, checkpoint)
│   ├── utils/                   # Utilities (seed, device)
│   └── registry.py              # Central registry system
├── configs/
│   ├── model/                   # Model configs (mlp.yaml, resnet20.yaml)
│   ├── training/                # Training configs (standard.yaml, bnn_sgld.yaml)
│   ├── dataset/                 # Dataset configs (mnist.yaml, cifar10.yaml)
│   └── <example_configs>.yaml   # Example configs
├── data/                        # Downloaded datasets
├── outputs/                     # Training outputs and checkpoints
├── train.py                     # Main training script
└── README.md
```

## Installation

This project was developed and tested on **Python 3.10.19**. 

### 1. Clone the Repository

```bash
git clone https://github.com/jpweideman/BNN-EDL.git
cd BNN-EDL
```

### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies

```bash
# Configure Poetry to use Python 3.10 
# (If installed. Otherwise Poetry will use your system's default Python.)
poetry env use python3.10

# Install all dependencies
poetry install
```

### 4. **Activate the virtual environment**:
```bash
source $(poetry env info --path)/bin/activate
```

### 5. **Run training**:
```bash
# After activation, run commands normally
python train.py --config-name mnist_mlp.json

# Or use poetry run without activation
poetry run python train.py --config-name mnist_mlp.json
```

## Usage

### Resuming Training

Training automatically saves checkpoints every epoch. To resume an interrupted training run:

```bash
# Find the output directory of your interrupted run
# Example: outputs/2026-01-01/12-00-00/

# Resume training in the same directory
poetry run python train.py --config-name cifar10_resnet20 \
  hydra.run.dir=outputs/2026-01-01/12-00-00/
```




