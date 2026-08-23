# Bayesian Neural Networks for Classification

Uncertainty quantification for image classification: Bayesian neural networks sampled with MCMC via [posteriors](https://github.com/normal-computing/posteriors), Evidential Deep Learning via [edl-pytorch](https://github.com/teddykoker/evidential-learning-pytorch), and the combination of the two — a Dirichlet head sampled with SGLD under a function-space prior on the total concentration.

Every component is registered by name in `src/registry.py` and selected from a Hydra config, so adding one means dropping a module in the matching `src/` folder and naming it in a yaml.

## What is implemented

| | |
|---|---|
| Architectures | `mlp`, `resnet20` (filter response norm) |
| Output layers | `linear` (softmax logits), `dirichlet` (evidential concentrations) |
| Optimizers | `sgd`, `adam`, `adamw` |
| Schedulers | `cosine_annealing`, `step_lr`, `exponential_lr` |
| Samplers | `sgld`, `sglrw`, `sghmc` |
| Likelihoods | `categorical`, `dirichlet` |
| Losses | `cross_entropy`, `edl_log`, `edl_digamma`, `edl_mse` |
| Priors | `diagonal_normal` over the weights, `gamma_strength` over the Dirichlet total concentration |
| Datasets | `fashion_mnist`, `cifar10` |
| Transforms | `to_tensor`, `normalize`, `random_crop`, `random_horizontal_flip`, `flatten` |

### Metrics

Metrics are chosen per evaluation split by name. `src/metrics/` is the full list; the families are:

| Family | Names |
|---|---|
| Softmax | `accuracy`, `loss`, `nll`, `brier_score`, `calibration_error` |
| Dirichlet | `dirichlet_nll`, `dirichlet_digamma_nll`, `dirichlet_brier_score`, `dirichlet_expected_brier`, `dirichlet_calibration_error`, `dirichlet_strength`, `vacuity` |
| Uncertainty decomposition | `analytical_dirichlet_{total,aleatoric,distributional}_uncertainty` |
| Averaged over posterior samples | `bma_` counterparts of most of the above, plus `bma_[dirichlet_]{predictive_entropy,expected_entropy,mutual_information,predictive_variance}` |
| Per-input arrays | `array_dump`, which collects per-input values instead of a scalar |

## Configuration

Hydra composes each experiment config from four defaults groups:

```
defaults:
  - datasets: cifar10                 # configs/datasets/  — train/val/test loaders
  - model: resnet20                   # configs/model/     — architecture and output layer
  - training: standard                # configs/training/  — optimizer or sampler, loss, priors, W&B
  - evaluation: standard_cifar10      # configs/evaluation/— per-split intervals and metrics
  - _self_
```

Training defines either `optimizer` or `sampler`, not both. LR schedulers apply only to optimizer-based training and are ignored, with a warning, for samplers.

There is one experiment config per dataset × method, each carrying the protocol as its defaults — train on the train split, checkpoint on val, evaluate test only at the end — so an experiments yaml overrides only what it changes:

| Method | Fashion-MNIST | CIFAR-10 |
|---|---|---|
| SGD softmax | `fashion_mnist_sgd` | `cifar10_sgd` |
| EDL | `fashion_mnist_edl` | `cifar10_edl` |
| Categorical BNN | `fashion_mnist_categorical_bnn_sgld` | `cifar10_categorical_bnn_sgld` |
| eBNN (Dirichlet BNN) | `fashion_mnist_dirichlet_bnn_sgld` | `cifar10_dirichlet_bnn_sgld` |

Each pairs with the matching evaluation preset (`standard_*`, `edl_*`, `bnn_*`, `dirichlet_bnn_*`). Evaluation intervals are `1`/`N` for every epoch or every N, `-1` for the final epoch only, `0` to disable.

```bash
python train.py --config-name cifar10_categorical_bnn_sgld training.sampler.params.temperature=0.01
python train.py --config-name fashion_mnist_dirichlet_bnn_sgld evaluation.fashion_mnist_val.interval=5
```

## Repository structure

```
BNN-EDL/
├── configs/
│   ├── datasets/ model/ training/ evaluation/    # Defaults groups
│   └── *.yaml                                    # Experiment configs (dataset x method)
├── src/
│   ├── models/ optimizers/ samplers/ losses/ likelihoods/
│   ├── priors/ priors_fs/ schedulers/ metrics/ data/
│   ├── training/          # Engines, evaluators, handlers, checkpointing
│   ├── builders/          # Config -> component, via the registry
│   ├── utils/
│   └── registry.py
├── analysis/              # Scripts that produce every reported number
├── results/               # Their outputs: csv files and LaTeX tables
├── tests/
├── experiments_*.yaml     # Experiment lists for run_experiments.py
├── train.py
├── run_experiments.py
└── outputs/               # One directory per run
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
# or
wget -qO- https://install.python-poetry.org | python3 -
```

### 3. Ensure Poetry is on PATH

If `poetry --version` fails with `command not found` after installation, add Poetry's bin directory to your shell `PATH`, then reload your shell configuration and run:

```bash
poetry --version
```

### 4. Install Python 3.10 with pyenv

Install pyenv for your OS first, then run:

```bash
pyenv install 3.10.19
pyenv local 3.10.19
poetry env use "$(pyenv which python)"
```

### 5. Install Dependencies

```bash
poetry install
```

### 6. **Activate the virtual environment**:
```bash
source $(poetry env info --path)/bin/activate
```

### 7. **Run training**:
```bash
# After activation, run commands normally
python train.py --config-name fashion_mnist_sgd

# Or use poetry run without activation
poetry run python train.py --config-name fashion_mnist_sgd
```

## Usage

### Run outputs

Each run writes one directory, `outputs/<run>/`:

```
.hydra/config.yaml     The composed config the run actually used
best_model.pt          Best checkpoint on the checkpoint split, with its score
last_checkpoint.pt     Written every epoch, for resuming
samples/               Posterior snapshots (sampled runs)
arrays/<split>.npz     Per-input arrays, for every split with an array_dump metric
arrays/summary.json    Each split's final metrics, readable without W&B
metrics.json           The W&B run summary
```

W&B metric names are `<split>/<metric>`, so training and each evaluation split get their own section.

### Resuming a run

Checkpoints are written every epoch. Point a run at its own output directory to continue it:

```bash
python train.py --config-name cifar10_sgd hydra.run.dir=outputs/2026-01-01/12-00-00/
```

A resumed run keeps the restored weights: `training.pretrained` is applied only on a fresh start.

### Running experiment sets

`run_experiments.py` runs the entries of an experiments yaml in declaration order, once per seed, into `outputs/<entry>_s<seed>/`. Completed runs are recorded in `.<experiments file>_state.json`, so an interrupted set resumes where it stopped.

```bash
python run_experiments.py --file_name experiments_e1.yaml --runs 3
python run_experiments.py --file_name experiments_e1.yaml --list
```

`--only`, `--skip` and `--rerun` take entry names. An entry can declare `pretrained_from` to warm-start from another entry's best checkpoint, of the same seed, or of the next seed with `pretrained_from_next_seed`, which the runner defers to a second pass once every seed's source run exists.

### Warm starts

`training.pretrained` loads another run's checkpoint before training. For a Dirichlet head under a `gamma_strength` prior, `training.pretrained.match_prior_mode=true` also adds one constant to the output bias, so the pretrained model's median total concentration starts at the prior mode without changing which class it predicts.
