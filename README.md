# Bayesian Neural Networks for Classification

Bayesian Neural Networks (BNNs) via [posteriors](https://github.com/normal-computing/posteriors) and Evidential Deep Learning (EDL) via [edl-pytorch](https://github.com/teddykoker/evidential-learning-pytorch) for uncertainty quantification in classification.

## Features

### Uncertainty methods

- **Standard training**: Adam / SGD with optional LR scheduler
- **BNN**: SGLD, SGLRW, SGHMC, Gibbs SGLD with categorical or Dirichlet likelihood
- **EDL**: Dirichlet output layer with digamma, log, or MSE loss
- **Dirichlet BNN**: Dirichlet output + MCMC sampling + optional function-space prior on total concentration

### Architectures and outputs

- **MLP**, **ResNet20** (FRN)
- **Linear** output (softmax logits)
- **Dirichlet** output (evidential concentrations)

### Datasets

- **Fashion-MNIST** (28×28 grayscale)
- **CIFAR-10** (32×32 RGB)

### Metrics

**Standard / softmax:** `accuracy`, `loss`, `nll`, `brier_score`, `calibration_error`

**BMA (ensemble over posterior samples):** `bma_accuracy`, `bma_nll`, `bma_brier_score`, `bma_calibration_error`, `bma_predictive_entropy`, `bma_expected_entropy`, `bma_mutual_information`, `bma_predictive_variance`

**Dirichlet (mean probability α/S):** `dirichlet_nll`, `dirichlet_brier_score`, `dirichlet_calibration_error`, `dirichlet_strength`, `vacuity`

**Dirichlet (full distribution):** `dirichlet_digamma_nll`, `dirichlet_expected_brier`

**Dirichlet BMA:** `bma_dirichlet_*` counterparts for accuracy, NLL, Brier, calibration, and uncertainty decomposition

**Analytical Dirichlet decomposition:** `analytical_dirichlet_total_uncertainty`, `analytical_dirichlet_aleatoric_uncertainty`, `analytical_dirichlet_distributional_uncertainty`

## Configuration

Hydra composes experiment configs from defaults groups under `configs/`:

```
defaults:
  - datasets: cifar10
  - model: resnet20
  - training: standard          # or bnn_sgld, edl, dirichlet_bnn_sgld, ...
  - evaluation: standard_cifar10_test
  - _self_
```

| Group | Path | Role |
|-------|------|------|
| `datasets` | `configs/datasets/` | Train/test/OOD dataloaders |
| `model` | `configs/model/` | Architecture and output layer |
| `training` | `configs/training/` | Optimizer or sampler, loss, checkpointing, W&B |
| `evaluation` | `configs/evaluation/` | Per-dataset eval intervals and metrics |

Experiment configs in `configs/` and `configs/templates/` override group defaults. Training uses either `optimizer` or `sampler`, not both. LR schedulers apply only to optimizer-based training; they are ignored (with a warning) for sampler-based runs.

### Evaluation presets

| Preset | Use case |
|--------|----------|
| `standard_cifar10_test` / `standard_fashion_mnist_test` | Deterministic softmax baseline |
| `bnn_cifar10_test` / `bnn_fashion_mnist` | Categorical BNN + BMA metrics |
| `edl_cifar10_test` / `edl_fashion_mnist` | EDL single-model + analytical decomposition |
| `dirichlet_bnn_cifar10_test` / `dirichlet_bnn_fashion_mnist` | Dirichlet BNN + full metric suite |

Override at runtime:

```bash
python train.py --config-name cifar10_resnet20_bnn_sgld_T_0.01 evaluation=bnn_cifar10_test
python train.py --config-name fashion_mnist_dirichlet_bnn_sgld evaluation.dirichlet_bnn_fashion_mnist.interval=5
```

## Repository structure

```
BNN-EDL/
├── configs/
│   ├── datasets/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   ├── templates/
│   └── *.yaml                 # Experiment configs
├── src/
│   ├── models/
│   ├── optimizers/
│   ├── samplers/
│   ├── losses/
│   ├── likelihoods/
│   ├── priors/
│   ├── priors_fs/
│   ├── schedulers/
│   ├── metrics/
│   ├── data/
│   ├── training/
│   ├── builders/
│   └── registry.py
├── train.py
└── outputs/
```

## Installation

Developed on **Python 3.10**.

```bash
git clone https://github.com/jpweideman/BNN-EDL.git
cd BNN-EDL
poetry install
source $(poetry env info --path)/bin/activate
```

See [Poetry](https://python-poetry.org/docs/#installation) and [pyenv](https://github.com/pyenv/pyenv) docs if needed. Pin Python with `poetry env use 3.10`.

## Usage

```bash
# Deterministic CIFAR-10 baseline
python train.py --config-name cifar10_resnet20

# Categorical BNN (SGLD temperature sweep)
python train.py --config-name cifar10_resnet20_bnn_sgld_T_0.01

# EDL
python train.py --config-name edl_cifar10_resnet20_digamma_loss

# Dirichlet BNN with function-space prior
python train.py --config-name fashion_mnist_dirichlet_bnn_sgld

# Multirun (e.g. temperature grid)
python train.py --config-name cifar10_resnet20_bnn_sgld_T_0.01 --multirun \
  training.sampler.params.temperature=0.001,0.01,0.1,1.0
```

### Resuming training

Checkpoints are saved each epoch. Resume in the same output directory:

```bash
python train.py --config-name cifar10_resnet20 \
  hydra.run.dir=outputs/2026-01-01/12-00-00/
```
